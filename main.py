"""
backend/main.py
FastAPI — PDF Editor Pro Web
"""
import os
import uuid
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="PDF Editor Pro API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/tmp/quillr_sessions"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def session_path(session_id: str) -> Path:
    p = UPLOAD_DIR / session_id
    p.mkdir(exist_ok=True)
    return p

def get_pdf_path(session_id: str) -> Path:
    return session_path(session_id) / "document.pdf"

class EraseRequest(BaseModel):
    session_id: str
    page: int
    x_pct: float
    y_pct: float
    w_pct: float
    h_pct: float

class TextEditRequest(BaseModel):
    session_id: str
    block_id: str
    new_text: str

class SaveRequest(BaseModel):
    session_id: str

@app.get("/")
def root():
    return {"status": "PDF Editor Pro API online"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Apenas arquivos PDF são aceitos.")

    session_id = str(uuid.uuid4())
    pdf_path = get_pdf_path(session_id)

    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    import fitz
    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    pages_info = []
    for i in range(page_count):
        r = doc[i].rect
        pages_info.append({"width": r.width, "height": r.height})
    doc.close()

    return {
        "session_id": session_id,
        "filename": file.filename,
        "page_count": page_count,
        "pages": pages_info,
    }

@app.get("/render/{session_id}/{page}")
async def render_page(session_id: str, page: int, zoom: float = 1.5):
    pdf_path = get_pdf_path(session_id)
    if not pdf_path.exists():
        raise HTTPException(404, "Sessão não encontrada.")

    import fitz
    doc = fitz.open(str(pdf_path))
    if page < 0 or page >= doc.page_count:
        raise HTTPException(400, "Página inválida.")

    pix = doc[page].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img_path = session_path(session_id) / f"page_{page}_{uuid.uuid4().hex[:8]}.png"
    pix.save(str(img_path))
    doc.close()

    return FileResponse(str(img_path), media_type="image/png")

@app.post("/extract/{session_id}/{page}")
async def extract_text(session_id: str, page: int):
    pdf_path = get_pdf_path(session_id)
    if not pdf_path.exists():
        raise HTTPException(404, "Sessão não encontrada.")

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from core.extractor import PDFExtractor

    groq_key = os.environ.get("GROQ_API_KEY", "")
    extractor = PDFExtractor(groq_api_key=groq_key)
    extractor.open(str(pdf_path))
    blocks = extractor.extract_page(page)

    result = []
    for b in blocks:
        result.append({
            "id": b.id,
            "text": b.text,
            "x0": b.x0, "y0": b.y0,
            "x1": b.x1, "y1": b.y1,
            "font_size": b.font_size,
            "font_name": b.font_name,
            "is_bold": b.is_bold,
            "is_italic": b.is_italic,
            "color_rgb": list(b.color_rgb),
            "align": b.align,
            "source": b.source,
        })
    extractor.close()
    return {"blocks": result}

@app.post("/erase")
async def erase_area(req: EraseRequest):
    """
    Apaga área com PyMuPDF redaction.
    NAO converte a pagina pra imagem — preserva texto nativo para continuar editando.
    """
    pdf_path = get_pdf_path(req.session_id)
    if not pdf_path.exists():
        raise HTTPException(404, "Sessão não encontrada.")

    try:
        import fitz
        from PIL import Image as PILImage

        doc = fitz.open(str(pdf_path))
        page = doc[req.page]
        pw, ph = page.rect.width, page.rect.height

        # Converte % → pontos PDF
        x0 = req.x_pct / 100 * pw
        y0 = req.y_pct / 100 * ph
        x1 = (req.x_pct + req.w_pct) / 100 * pw
        y1 = (req.y_pct + req.h_pct) / 100 * ph
        erase_rect = fitz.Rect(x0, y0, x1, y1)

        # Detecta cor do fundo nas bordas da área
        margin = 12
        sample_rect = fitz.Rect(
            max(0, x0 - margin), max(0, y0 - margin),
            min(pw, x1 + margin), min(ph, y1 + margin)
        )
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=sample_rect, alpha=False)
        img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
        w, h = img.size
        margin_px = max(6, int(margin * 2))

        border_pixels = []
        for x in range(w):
            for y in list(range(min(margin_px, h))) + list(range(max(0, h - margin_px), h)):
                border_pixels.append(img.getpixel((x, y)))
        for y in range(h):
            for x in list(range(min(margin_px, w))) + list(range(max(0, w - margin_px), w)):
                border_pixels.append(img.getpixel((x, y)))

        if border_pixels:
            r = sum(p[0] for p in border_pixels) / len(border_pixels) / 255
            g = sum(p[1] for p in border_pixels) / len(border_pixels) / 255
            b = sum(p[2] for p in border_pixels) / len(border_pixels) / 255
            fill_color = (r, g, b)
        else:
            fill_color = (1.0, 1.0, 1.0)

        # Redaction — remove conteudo SEM converter pagina pra imagem
        page.add_redact_annot(erase_rect, fill=fill_color)
        try:
            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_REMOVE,
                graphics=fitz.PDF_REDACT_LINE_NONE,
            )
        except Exception:
            try:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_REMOVE)
            except Exception:
                page.apply_redactions()

        tmp_pdf = str(pdf_path) + ".tmp"
        doc.save(tmp_pdf, garbage=4, deflate=True)
        doc.close()
        os.replace(tmp_pdf, str(pdf_path))

        return {"ok": True, "message": "Área apagada. Texto preservado."}

    except Exception as e:
        raise HTTPException(500, f"Erro ao apagar: {e}")

@app.post("/signature")
async def add_signature(
    session_id: str = Form(...),
    page: int = Form(...),
    x_pct: float = Form(...),
    y_pct: float = Form(...),
    w_pct: float = Form(...),
    h_pct: float = Form(...),
    file: UploadFile = File(...),
):
    pdf_path = get_pdf_path(session_id)
    if not pdf_path.exists():
        raise HTTPException(404, "Sessão não encontrada.")

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from core.inpainting_engine import pdf_page_to_image
        import fitz
        import numpy as np
        from PIL import Image as PILImage

        sig_path = session_path(session_id) / f"sig_{uuid.uuid4()}.png"
        with open(sig_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        dpi = 200
        img = pdf_page_to_image(str(pdf_path), page, dpi=dpi)
        ih, iw = img.shape[:2]

        x1 = max(0,  int(x_pct / 100 * iw))
        y1 = max(0,  int(y_pct / 100 * ih))
        w  = max(10, int(w_pct / 100 * iw))
        h  = max(10, int(h_pct / 100 * ih))

        sig = PILImage.open(str(sig_path)).convert("RGBA")
        sig = sig.resize((w, h), PILImage.LANCZOS)
        base = PILImage.fromarray(img[:, :, ::-1])
        base.paste(sig, (x1, y1), sig.split()[3])

        img_result = np.array(base)[:, :, ::-1]

        try:
            import cv2
            rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = img_result[:, :, ::-1]
        pil_result = PILImage.fromarray(rgb)

        doc = fitz.open(str(pdf_path))
        tmp_page = str(session_path(session_id) / f"tmp_sig_{page}.pdf")
        pil_result.save(tmp_page, format="PDF", resolution=dpi)
        tmp_doc = fitz.open(tmp_page)
        doc.delete_page(page)
        doc.insert_pdf(tmp_doc, from_page=0, to_page=0, start_at=page)
        tmp_doc.close()
        os.remove(tmp_page)
        os.remove(str(sig_path))

        tmp_pdf = str(pdf_path) + ".tmp"
        doc.save(tmp_pdf, garbage=4, deflate=True)
        doc.close()
        os.replace(tmp_pdf, str(pdf_path))

        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, f"Erro ao inserir assinatura: {e}")

@app.post("/save-text")
async def save_text_edits(
    session_id: str = Form(...),
    edits: str = Form(...),
):
    import json
    pdf_path = get_pdf_path(session_id)
    if not pdf_path.exists():
        raise HTTPException(404, "Sessão não encontrada.")

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from core.inpainting_engine import (
            pdf_all_pages_to_images, remove_content_inpaint,
            smart_replace_text, image_to_pdf)

        edits_list = json.loads(edits)
        dpi = 200
        all_imgs = pdf_all_pages_to_images(str(pdf_path), dpi=dpi)

        import fitz
        doc = fitz.open(str(pdf_path))

        by_page = {}
        for e in edits_list:
            by_page.setdefault(e["page"], []).append(e)

        for page_idx, page_edits in by_page.items():
            img = all_imgs[page_idx].copy()
            ih, iw = img.shape[:2]
            page = doc[page_idx]
            pw, ph = page.rect.width, page.rect.height
            sx, sy = iw / pw, ih / ph

            for edit in page_edits:
                x1 = max(0,  int(edit["x0"] * sx))
                y1 = max(0,  int(edit["y0"] * sy))
                x2 = min(iw, int(edit["x1"] * sx))
                y2 = min(ih, int(edit["y1"] * sy))
                img = remove_content_inpaint(
                    img, x1, y1, x2, y2,
                    threshold=80, full_area=False, radius=5)
                r, g, b = edit.get("color_rgb", [0, 0, 0])
                img = smart_replace_text(
                    img, edit["new_text"], x1, y1, x2, y2,
                    original_text=edit["original_text"],
                    fontname_hint=edit.get("font_name", "arial"),
                    font_size_hint=max(8, int((y2 - y1) * 0.80)),
                    color_bgr=(int(b*255), int(g*255), int(r*255)),
                    align=edit.get("align", "left"))
            all_imgs[page_idx] = img

        doc.close()
        tmp = str(pdf_path) + ".tmp"
        image_to_pdf(all_imgs, tmp, dpi=dpi)
        os.replace(tmp, str(pdf_path))

        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, f"Erro ao salvar texto: {e}")

@app.get("/download/{session_id}")
async def download_pdf(session_id: str):
    pdf_path = get_pdf_path(session_id)
    if not pdf_path.exists():
        raise HTTPException(404, "Sessão não encontrada.")
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename="documento_editado.pdf",
    )

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    p = session_path(session_id)
    if p.exists():
        shutil.rmtree(str(p))
    return {"ok": True}
