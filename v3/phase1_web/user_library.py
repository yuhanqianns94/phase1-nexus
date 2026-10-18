"""Private, bounded document uploads with explicit user-owned API credentials.

Only the caller-provided session workspace is writable. Credentials travel in
memory to the worker and are never included in extraction artifacts or state
files. The ordinary local extraction CLI keeps its own client defaults.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit
from xml.parsers import expat

import fitz
from openai import OpenAI

from phase1_extraction.llm import BudgetGuard
from phase1_extraction.pipeline import run_pipeline
from phase1_extraction.request_cache import artifact_lock

MAX_FILES = 10
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_TOTAL_BYTES = 50 * 1024 * 1024
MAX_XML_BYTES = 2 * 1024 * 1024
MAX_PAGES_PER_FILE = 80
MAX_TOTAL_PAGES = 150
MAX_STORED_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_STORED_UPLOADS = 30
MAX_RENDER_PIXELS = 18_000_000
MAX_XML_ELEMENTS = 50_000
MAX_XML_DEPTH = 64
MAX_PARSED_ASSET_BYTES = 64 * 1024 * 1024
_STAGED_NAME = re.compile(r"[a-f0-9]{64}\.(pdf|xml)\Z")
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}\Z")


class UserLibraryError(ValueError):
    """A safe, fixed English message that may be shown in the public UI."""


def _safe_directory(workspace: Path | str, name: str) -> Path:
    root = Path(workspace).resolve()
    path = root / name
    if path.is_symlink() or path.resolve().parent != root:
        raise UserLibraryError("The private library workspace is unavailable.")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def _validate_xml(data: bytes) -> None:
    if len(data) > MAX_XML_BYTES:
        raise UserLibraryError("Each XML document must be at most 2 MB.")
    parser = expat.ParserCreate(namespace_separator="}")
    depth = count = table_cells = expanded_cells = 0

    def reject(*_args: Any) -> None:
        raise UserLibraryError(
            "XML must be standalone JATS without DTDs, entities, includes or processing instructions. "
            "Upload a standalone XML file or the article PDF."
        )

    def start(name: str, attrs: dict[str, str]) -> None:
        nonlocal depth, count, table_cells, expanded_cells
        depth += 1
        count += 1
        tag = name.rsplit("}", 1)[-1]
        if count == 1 and tag != "article":
            raise UserLibraryError("XML uploads must contain a single JATS article.")
        if name.startswith("http://www.w3.org/2001/XInclude}"):
            reject()
        if depth > MAX_XML_DEPTH or count > MAX_XML_ELEMENTS:
            raise UserLibraryError("This XML document exceeds the supported complexity limit.")
        for attr, value in attrs.items():
            attribute = attr.rsplit("}", 1)[-1]
            # The trusted local JATS parser intentionally resolves graphic hrefs
            # relative to its XML file. Public uploads must never reach that
            # branch with a file/resource reference, including absolute paths.
            if attribute.casefold() in {"href", "src", "data", "poster", "base", "schemalocation", "nonamespaceschemalocation"}:
                try:
                    link = urlsplit(value)
                    bibliographic_link = (tag == "ext-link" and attribute == "href"
                                          and link.scheme in {"http", "https"} and bool(link.netloc)
                                          and link.username is None and link.password is None)
                except ValueError:
                    bibliographic_link = False
                if not bibliographic_link:
                    raise UserLibraryError(
                        "XML uploads cannot reference image, media or local files. "
                        "Only HTTP(S) bibliographic ext-link URLs are supported. Upload the article PDF when figures are required."
                    )
            if attribute in {"rowspan", "colspan", "morerows"}:
                if not value.isdecimal() or len(value) > 3 or int(value) > 100:
                    raise UserLibraryError("A table in this XML document exceeds the supported complexity limit.")
        if tag in {"td", "th"}:
            table_cells += 1
            expanded_cells += max(1, int(attrs.get("rowspan", "1"))) * max(1, int(attrs.get("colspan", "1")))
            if table_cells > 4000 or expanded_cells > 20_000:
                raise UserLibraryError("The XML tables exceed the supported complexity limit.")

    def end(_name: str) -> None:
        nonlocal depth
        depth -= 1

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.StartDoctypeDeclHandler = reject
    parser.EntityDeclHandler = reject
    parser.ExternalEntityRefHandler = reject
    parser.ProcessingInstructionHandler = reject
    try:
        parser.Parse(data, True)
    except UserLibraryError:
        raise
    except expat.ExpatError:
        raise UserLibraryError("The XML document is invalid or incomplete.") from None


def _validate_pdf(data: bytes) -> int:
    if not data.startswith(b"%PDF-"):
        raise UserLibraryError("The uploaded file is not a valid PDF.")
    try:
        with fitz.open(stream=data, filetype="pdf") as document:
            if document.needs_pass:
                raise UserLibraryError("Password-protected PDFs are not supported.")
            pages = document.page_count
            if not 1 <= pages <= MAX_PAGES_PER_FILE:
                raise UserLibraryError("Each PDF must contain 1 to 80 pages.")
            for page in document:
                width, height = page.rect.width, page.rect.height
                if (not math.isfinite(width * height) or width <= 0 or height <= 0
                        or width * height * (200 / 72) ** 2 > MAX_RENDER_PIXELS):
                    raise UserLibraryError("A PDF page exceeds the supported rendering size.")
                if any(image[2] * image[3] > 40_000_000 for image in page.get_images()):
                    raise UserLibraryError("A PDF image exceeds the supported rendering size.")
            return pages
    except UserLibraryError:
        raise
    except Exception:
        raise UserLibraryError("The PDF could not be opened. Please upload a valid article PDF.") from None


def _validated_files(files: Iterable[tuple[str, bytes]]) -> list[tuple[str, bytes]]:
    files = list(files)
    if not 1 <= len(files) <= MAX_FILES:
        raise UserLibraryError("Upload 1 to 10 PDF or XML files at a time.")
    total_bytes = sum(len(data) for _, data in files)
    if total_bytes > MAX_TOTAL_BYTES:
        raise UserLibraryError("The total upload size must be at most 50 MB.")
    validated: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    total_pages = 0
    for name, data in files:
        suffix = Path(name).suffix.lower()
        if suffix not in {".pdf", ".xml"}:
            raise UserLibraryError("Only PDF and JATS XML documents are supported.")
        if not data or len(data) > MAX_FILE_BYTES:
            raise UserLibraryError("Each upload must be non-empty and at most 20 MB.")
        staged_name = hashlib.sha256(data).hexdigest() + suffix
        if staged_name in seen:
            continue
        seen.add(staged_name)
        if suffix == ".xml":
            _validate_xml(data)
        else:
            total_pages += _validate_pdf(data)
        if total_pages > MAX_TOTAL_PAGES:
            raise UserLibraryError("A batch may contain at most 150 PDF pages in total.")
        validated.append((staged_name, data))
    return validated


def stage_uploaded_documents(workspace: Path | str, files: Iterable[tuple[str, bytes]]) -> list[str]:
    """Validate before writing; return private content-addressed relative paths."""
    validated = _validated_files(files)
    upload_dir = _safe_directory(workspace, "uploads")
    with artifact_lock(upload_dir / ".upload.lock"):
        existing = {path.name: path.stat().st_size for path in upload_dir.iterdir()
                    if _STAGED_NAME.fullmatch(path.name) and not path.is_symlink()}
        prospective = {**existing, **{name: len(data) for name, data in validated}}
        if len(prospective) > MAX_STORED_UPLOADS or sum(prospective.values()) > MAX_STORED_UPLOAD_BYTES:
            raise UserLibraryError("This session has reached its private upload limit (30 files or 100 MB).")
        for name, data in validated:
            destination = upload_dir / name
            if destination.is_symlink():
                raise UserLibraryError("The private library workspace is unavailable.")
            # Never use a user-supplied filename as a filesystem destination.
            destination.write_bytes(data)
            destination.chmod(0o600)
    return [f"uploads/{name}" for name, _ in validated]


def _load_staged_documents(workspace: Path | str, document_paths: list[str]) -> list[Path]:
    if not 1 <= len(document_paths) <= MAX_FILES:
        raise UserLibraryError("Upload 1 to 10 PDF or XML files at a time.")
    root = Path(workspace).resolve()
    uploads = _safe_directory(root, "uploads")
    files: list[tuple[str, bytes]] = []
    for relative in document_paths:
        item = Path(relative)
        if len(item.parts) != 2 or item.parts[0] != "uploads" or not _STAGED_NAME.fullmatch(item.name):
            raise UserLibraryError("The selected private document is unavailable.")
        path = root / item
        if path.is_symlink() or path.resolve().parent != uploads or not path.is_file():
            raise UserLibraryError("The selected private document is unavailable.")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise UserLibraryError("Each upload must be non-empty and at most 20 MB.")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != path.stem:
            raise UserLibraryError("The selected document changed. Please upload it again.")
        files.append((path.name, data))
    validated = _validated_files(files)
    return [uploads / name for name, _ in validated]


def _safe_failure(error: Exception) -> str:
    if isinstance(error, UserLibraryError):
        return str(error)
    kind = type(error).__name__
    if kind in {"AuthenticationError", "PermissionDeniedError"}:
        return "The API provider rejected access. Check your key and model permissions."
    if kind == "RateLimitError":
        return "The API provider reported a rate or quota limit. Check your account before retrying."
    if kind in {"APITimeoutError", "APIConnectionError"}:
        return "The API provider could not be reached. Please retry later."
    if isinstance(error, RuntimeError) and "Conservative request reserve" in str(error):
        return "The next request exceeds your remaining estimated budget. Adjust the budget or token limit."
    if isinstance(error, ValueError) and str(error) == "Parsed assets exceed the upload processing limit.":
        return "The rendered article is too large to process on this website. Upload a smaller PDF or its standalone JATS XML."
    return "Extraction could not finish. Check your model, token limit, document and API account before retrying."


def extract_uploaded_documents(
    *, workspace: Path | str, document_paths: list[str], api_key: str, model: str,
    reasoning_effort: str, max_output_tokens: int, max_usd: float,
    input_usd_per_million: float, output_usd_per_million: float,
    group_as_article: bool = False,
) -> dict[str, Any]:
    """Worker entry point; all paid requests use the supplied key and model.

    A successful result is immediately consumable under ``workspace/library``.
    Identical requests resume the existing paid-response cache in this session.
    Exceptions shown to users or a job service never contain provider details.
    """
    if not isinstance(api_key, str) or not 8 <= len(api_key.strip()) <= 512 or any(c.isspace() for c in api_key.strip()):
        raise UserLibraryError("Enter your own OpenAI API key to extract uploaded documents.")
    api_key = api_key.strip()
    model = model.strip()
    if not _MODEL_ID.fullmatch(model) or model.lower().startswith("sk-") or model == api_key:
        raise UserLibraryError("Enter a valid model ID from your API provider.")
    if reasoning_effort not in {"", "none", "minimal", "low", "medium", "high"}:
        raise UserLibraryError("Choose a supported reasoning effort.")
    if not 1024 <= max_output_tokens <= 48000:
        raise UserLibraryError("Set an output token limit between 1,024 and 48,000.")
    values = (max_usd, input_usd_per_million, output_usd_per_million)
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise UserLibraryError("Enter a positive budget and the current input and output token prices.")
    sources = _load_staged_documents(workspace, document_paths)
    library = _safe_directory(workspace, "library")
    guard = BudgetGuard(max_usd=max_usd, input_usd_per_million=input_usd_per_million,
                        output_usd_per_million=output_usd_per_million,
                        cached_input_multiplier=1.0)
    groups = [sources] if group_as_article else [[source] for source in sources]
    result: dict[str, Any] = {"completed": [], "failed": [], "estimated_usd": 0.0}
    client = None
    try:
        # Explicit URL and credentials prevent host environment/provider defaults
        # from redirecting a visitor's key or charging the owner's account.
        client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1",
                        organization="", project="", max_retries=0, timeout=120.0)
        for group in groups:
            identity = "|".join(sorted(path.name for path in group))
            document_id = "USER_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
            if (library / document_id).is_symlink():
                raise UserLibraryError("The private library workspace is unavailable.")
            try:
                run_pipeline(document_id=document_id, source_paths=group, output_root=library,
                             execute_model=True, model=model, reasoning_effort=reasoning_effort,
                             max_output_tokens=max_output_tokens, budget_guard=guard,
                             client=client, sanitize_errors=True, resume=True,
                             max_parsed_asset_bytes=MAX_PARSED_ASSET_BYTES,
                             material_roles=({str(path.resolve()): "main" if index == 0 else "supplement"
                                              for index, path in enumerate(group)} if group_as_article else None))
                result["completed"].append(document_id)
            except Exception as error:
                result["failed"].append({"document_id": document_id, "message": _safe_failure(error)})
    except Exception as error:
        raise UserLibraryError(_safe_failure(error)) from None
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        api_key = ""
    result["estimated_usd"] = guard.spent_usd
    return result


def render_user_library(st: Any, workspace: Path) -> None:
    """Render an opt-in uploader; the built-in library requires no API key."""
    st.subheader("Your private literature library")
    st.caption("Optional: upload articles to create a library for this browser session. "
               "Published shared evidence can be used without an API key. Private uploads are temporary.")
    existing = list((workspace / "library").glob("*/resolved_extraction.json"))
    if existing:
        st.success(f"{len(existing)} extracted article(s) are available in your private library.")
    uploaded = st.file_uploader("Upload article PDFs or standalone JATS XML", type=["pdf", "xml"],
                                accept_multiple_files=True, key="user_library_uploads",
                                help="Up to 10 files, 20 MB each (2 MB for XML), 50 MB total. PDFs: up to 80 pages each and 150 pages per batch.")
    with st.form("user_library_extraction"):
        api_key = st.text_input("Your OpenAI API key", type="password", key="user_library_api_key",
                               help="Used only for your extraction requests. The website owner's API key is never used.")
        choice = st.selectbox("Extraction model", ["Custom model ID", "gpt-4.1", "gpt-4.1-mini", "gpt-5"])
        custom = st.text_input("Custom model ID", placeholder="Enter the exact model ID from your API account")
        model = custom if choice == "Custom model ID" else choice
        effort = st.selectbox("Reasoning effort", ["Do not send (model default)", "none", "minimal", "low", "medium", "high"])
        tokens = st.number_input("Maximum output tokens per request", min_value=1024, max_value=48000, value=8192, step=1024)
        budget = st.number_input("Maximum estimated cost for this extraction (USD)", min_value=0.01, value=1.0, step=0.50)
        input_price = st.number_input("Current input price (USD per 1 million tokens)", min_value=0.0, value=0.0, step=0.1, format="%.4f")
        output_price = st.number_input("Current output price (USD per 1 million tokens)", min_value=0.0, value=0.0, step=0.1, format="%.4f")
        st.caption("Enter the current prices for your selected model. The budget is an estimate, not a provider billing cap. "
                   "Configure provider spending limits for additional control. The model must support images and structured JSON output.")
        grouped = st.checkbox("These files belong to one article (first file: main paper; remaining files: supplements)")
        accepted = st.checkbox("I have permission to process these documents and agree to send them to OpenAI using my key and pay the API charges.")
        submitted = st.form_submit_button("Extract my uploaded documents", type="primary")
    if submitted:
        if not accepted:
            st.error("Confirm document processing and API charges before starting.")
            return
        if not api_key.strip():
            st.error("Enter your own OpenAI API key to extract uploaded documents.")
            return
        if not uploaded:
            st.error("Upload at least one article first.")
            return
        if not _MODEL_ID.fullmatch(model.strip()) or model.strip().lower().startswith("sk-"):
            st.error("Enter a valid model ID from your API provider.")
            return
        if input_price <= 0 or output_price <= 0:
            st.error("Enter the current input and output token prices before starting.")
            return
        try:
            document_paths = stage_uploaded_documents(workspace, ((item.name, item.getvalue()) for item in uploaded))
            payload = {"document_paths": document_paths, "api_key": api_key,
                       "model": model, "reasoning_effort": "" if effort.startswith("Do not send") else effort,
                       "max_output_tokens": int(tokens), "max_usd": float(budget),
                       "input_usd_per_million": float(input_price), "output_usd_per_million": float(output_price),
                       "group_as_article": grouped}
            from phase1_web.jobs import run_task
            with st.spinner("Waiting for an available worker and extracting your documents…"):
                result = run_task("extract", payload, workspace, timeout=1800)
            st.session_state["user_library_result"] = result
        except Exception as error:
            st.error(_safe_failure(error))
        finally:
            api_key = ""
    result = st.session_state.get("user_library_result")
    if result:
        if result.get("completed"):
            st.success(f"Completed {len(result['completed'])} article(s). Your private library is ready to use.")
        for failure in result.get("failed", []):
            st.error(failure["message"])
        st.caption(f"Estimated charges for new requests: ${result.get('estimated_usd', 0.0):.4f}. "
                   "Matching completed requests in this private session are reused without a new API call.")
