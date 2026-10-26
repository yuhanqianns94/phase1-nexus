"""Open original article files with private-session boundaries."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import fitz

from phase1_web.user_library import MAX_FILE_BYTES, MAX_FILES, MAX_PAGES_PER_FILE


class ArticleFileUnavailable(ValueError):
    pass


def article_files(article_root, *, workspace=None):
    """Read saved material provenance; private sources must stay in this session."""
    root=Path(article_root)
    private=workspace is not None
    if private:
        workspace=Path(workspace).resolve()
        library=workspace/'library'
        uploads=workspace/'uploads'
        if (library.is_symlink() or uploads.is_symlink() or root.is_symlink()
                or root.resolve().parent!=library or library.resolve().parent!=workspace
                or uploads.resolve().parent!=workspace):
            raise ArticleFileUnavailable()
    manifest=root/'material_manifest.json'
    if not manifest.exists():
        manifest=root/'document_ir.json'
    try:
        if manifest.is_symlink() or manifest.stat().st_size>20*1024*1024:
            raise ArticleFileUnavailable()
        materials=json.loads(manifest.read_text())['materials']
        if not isinstance(materials,list) or (private and len(materials)>MAX_FILES):
            raise ArticleFileUnavailable()
        result=[]
        seen=set()
        for material in materials:
            raw=material.get('path') or material.get('local_path') or material.get('source_path')
            if not raw:
                continue
            path=Path(raw)
            if path.suffix.lower() not in {'.pdf','.xml'}:
                continue
            if private:
                # Public extraction records absolute paths to content-addressed uploads.
                if (not path.is_absolute() or path.is_symlink() or path.resolve().parent!=uploads
                        or not re.fullmatch(r'[a-f0-9]{64}\.(pdf|xml)',path.name)):
                    raise ArticleFileUnavailable()
                if material.get('sha256',path.stem)!=path.stem:
                    raise ArticleFileUnavailable()
            if not path.is_file() or (private and path.stat().st_size>MAX_FILE_BYTES):
                raise ArticleFileUnavailable()
            if path in seen:
                continue
            seen.add(path)
            result.append({'path':path,'role':material.get('role','main'),
                           'sha256':path.stem if private else material.get('sha256')})
        return sorted(result,key=lambda item:item['role']!='main')
    except (OSError,ValueError,KeyError,TypeError,AttributeError):
        raise ArticleFileUnavailable() from None


def read_article_file(item):
    data=item['path'].read_bytes()
    if item['sha256'] and hashlib.sha256(data).hexdigest()!=item['sha256']:
        raise ArticleFileUnavailable()
    return data


def pdf_page(data, page_number):
    """Render one bounded preview page, leaving the original download intact."""
    with fitz.open(stream=data,filetype='pdf') as document:
        if document.needs_pass or not 1<=document.page_count<=MAX_PAGES_PER_FILE:
            raise ArticleFileUnavailable()
        page=document[page_number-1]
        scale=min(2.0,1400/max(page.rect.width,page.rect.height))
        return page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False).tobytes('png')


def render_original_article(st, doc, article_root, *, library_source, workspace=None):
    if library_source=='shared':
        if re.fullmatch(r'PMC\d+',doc):
            st.markdown(f'[Open full article on PMC](https://pmc.ncbi.nlm.nih.gov/articles/{doc}/)')
        else:
            st.caption('No PMC full-text link is available for this article.')
        return
    private=library_source=='personal'
    unavailable=('The original uploaded file is unavailable in this session. Upload it again to view it.'
                 if private else 'The original article file is unavailable in the saved source location.')
    try:
        if article_root is None or (private and workspace is None):
            raise ArticleFileUnavailable()
        files=article_files(article_root,workspace=workspace if private else None)
        if not files:
            raise ArticleFileUnavailable()
    except ArticleFileUnavailable:
        st.info(unavailable)
        return
    title='Uploaded article' if private else 'Local article'

    @st.dialog(title,width='large')
    def show_article():
        try:
            # Recheck provenance and session ownership on every viewer interaction.
            current=article_files(article_root,workspace=workspace if private else None)
            if not current:
                raise ArticleFileUnavailable()
            selected=0
            if len(current)>1:
                def label(index):
                    item=current[index]
                    role='Main article' if item['role']=='main' else 'Supplement'
                    return f"{role} {index+1} ({item['path'].suffix[1:].upper()})"
                selected=st.selectbox('Article file',list(range(len(current))),format_func=label,
                                      key='article_file_'+doc)
            item=current[selected]
            data=read_article_file(item)
            st.caption(f"{title} · {item['path'].suffix[1:].upper()}")
            suffix=item['path'].suffix.lower()
            safe_doc=re.sub(r'[^A-Za-z0-9_-]','_',doc)
            st.download_button('Download original file',data,
                               file_name=f'{safe_doc}_{selected+1}{suffix}',
                               mime='application/pdf' if suffix=='.pdf' else 'application/xml',
                               key='article_download_'+doc,on_click='ignore')
            if suffix=='.pdf':
                with fitz.open(stream=data,filetype='pdf') as document:
                    count=document.page_count
                    if document.needs_pass or not 1<=count<=MAX_PAGES_PER_FILE:
                        raise ArticleFileUnavailable()
                number=st.number_input('Page',min_value=1,max_value=count,value=1,step=1,
                                       key=f'article_page_{doc}_{item["path"].stem}')
                st.caption(f'Page {number} of {count}')
                st.image(pdf_page(data,int(number)),width='stretch')
            else:
                # XML is shown as escaped source text; no document markup is executed.
                from phase1_web.user_library import _validate_xml
                if private:
                    _validate_xml(data)
                st.code(ET.tostring(ET.fromstring(data),encoding='unicode'),language='xml')
        except (ArticleFileUnavailable,OSError,ValueError,ET.ParseError,RuntimeError):
            st.info(unavailable)

    if st.button('Open uploaded article' if private else 'Open local article',key='open_article_'+doc):
        show_article()
