"""Original-file routing, private ownership, and offline document viewing."""
import json
from pathlib import Path

import fitz
import pytest
from streamlit.testing.v1 import AppTest

from phase1_extraction.pipeline import ingest
from phase1_web.article_source import ArticleFileUnavailable, article_files, read_article_file, pdf_page
from phase1_web.user_library import stage_uploaded_documents


def uploaded_article(workspace, *, supplements=False):
    with fitz.open() as document:
        for number in (1,2):
            page=document.new_page()
            page.insert_text((60,80),f'Uploaded article - original page {number}')
        data=document.tobytes()
    sources=[('paper.pdf',data)]
    if supplements:
        sources.append(('supplement.xml',b'<article><body><p>Original supplement text.</p></body></article>'))
    staged=stage_uploaded_documents(workspace,sources)
    # Even a PMC-shaped ID in the personal library must open the visitor's file.
    paths=[workspace/path for path in staged]
    article=ingest(document_id='PMC123',source_paths=paths,output_root=workspace/'library',
                   material_roles={str(path.resolve()):'main' if index==0 else 'supplement'
                                   for index,path in enumerate(paths)})
    return article.root,data


def source_app(workspace, article_root, source):
    import streamlit as st
    from phase1_web.evidence_review import _render_abstract, abstract_for
    _render_abstract(st,abstract_for(article_root),'PMC123',article_root=article_root,
                     library_source=source,workspace=workspace)


def test_personal_file_bytes_and_bounded_pdf_pages(tmp_path):
    root,data=uploaded_article(tmp_path)
    files=article_files(root,workspace=tmp_path)
    assert read_article_file(files[0])==data
    first=pdf_page(data,1)
    second=pdf_page(data,2)
    assert first!=second and first.startswith(b'\x89PNG')
    image=fitz.Pixmap(first)
    assert max(image.width,image.height)<=1401


@pytest.mark.parametrize('escape',['article_root','material_path','upload_symlink','manifest_symlink'])
def test_private_reader_cannot_open_another_session(tmp_path,escape):
    a,b=tmp_path/'a',tmp_path/'b'
    aroot,_=uploaded_article(a)
    broot,_=uploaded_article(b)
    manifest=aroot/'material_manifest.json'
    record=json.loads(manifest.read_text())
    if escape=='article_root':
        aroot=broot
    elif escape=='material_path':
        record['materials'][0]['path']=str(next((b/'uploads').glob('*.pdf')))
        manifest.write_text(json.dumps(record))
    elif escape=='upload_symlink':
        path=Path(record['materials'][0]['path'])
        path.unlink()
        path.symlink_to(next((b/'uploads').glob('*.pdf')))
    else:
        manifest.unlink()
        manifest.symlink_to(broot/'material_manifest.json')
    with pytest.raises(ArticleFileUnavailable):
        article_files(aroot,workspace=a)


def test_changed_or_deleted_original_is_unavailable(tmp_path):
    root,_=uploaded_article(tmp_path)
    files=article_files(root,workspace=tmp_path)
    files[0]['path'].write_bytes(b'changed')
    with pytest.raises(ArticleFileUnavailable):
        read_article_file(files[0])
    files[0]['path'].unlink()
    with pytest.raises(ArticleFileUnavailable):
        article_files(root,workspace=tmp_path)


def test_shared_library_has_pmc_link_without_private_file_access(tmp_path):
    app=AppTest.from_function(source_app,args=(str(tmp_path),str(tmp_path/'missing'),'shared')).run()
    assert not app.exception
    assert any('https://pmc.ncbi.nlm.nih.gov/articles/PMC123/' in item.value for item in app.markdown)
    assert not any(button.label=='Open uploaded article' for button in app.button)


def test_personal_library_opens_original_pdf_and_supplement(tmp_path):
    root,_=uploaded_article(tmp_path,supplements=True)
    app=AppTest.from_function(source_app,args=(str(tmp_path),str(root),'personal')).run()
    assert not any('pmc.ncbi.nlm.nih.gov' in item.value for item in app.markdown)
    next(button for button in app.button if button.label=='Open uploaded article').click().run()
    assert not app.exception
    assert any('Page 1 of 2' in item.value for item in app.caption)
    # AppTest reruns the full script, so reopen the dialog with the changed input.
    # The browser performs a dialog-only rerun for these controls.
    next(widget for widget in app.number_input if widget.label=='Page').set_value(2)
    next(button for button in app.button if button.label=='Open uploaded article').click().run()
    assert not app.exception
    assert any('Page 2 of 2' in item.value for item in app.caption)
    next(widget for widget in app.selectbox if widget.label=='Article file').set_value(1)
    next(button for button in app.button if button.label=='Open uploaded article').click().run()
    assert not app.exception
    assert any('Original supplement text.' in item.value for item in app.code)


def test_missing_personal_file_never_falls_back_to_pmc(tmp_path):
    app=AppTest.from_function(source_app,args=(str(tmp_path),str(tmp_path/'missing'),'personal')).run()
    assert not app.exception
    assert any('original uploaded file is unavailable' in item.value for item in app.info)
    assert not any('pmc.ncbi.nlm.nih.gov' in item.value for item in app.markdown)
    assert not app.button


def test_local_library_can_open_original_without_public_session(tmp_path):
    root,_=uploaded_article(tmp_path)
    app=AppTest.from_function(source_app,args=(None,str(root),'local')).run()
    next(button for button in app.button if button.label=='Open local article').click().run()
    assert not app.exception
    assert any('Page 1 of 2' in item.value for item in app.caption)
