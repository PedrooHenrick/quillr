"""
backend/main.py
FastAPI — PDF Editor Pro Web
"""
import os
import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

    extractor = PDFExtractor()
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


@app.get("/session-check/{session_id}")
async def session_check(session_id: str):
    pdf_path = get_pdf_path(session_id)
    if not pdf_path.exists():
        raise HTTPException(404, "Sessão não encontrada.")
    return {"ok": True}


@app.post("/erase")
async def erase_area(req: EraseRequest):
    """
    Apaga área usando inpainting do OpenCV — reconstrói o fundo real.
    Converte só a página afetada em imagem, aplica inpainting,
    reinsere no PDF e injeta camada de texto invisível para manter extração.
    """
    pdf_path = get_pdf_path(req.session_id)
    if not pdf_path.exists():
        raise HTTPException(404, "Sessão não encontrada.")

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from core.inpainting_engine import (
            pdf_page_to_image, remove_content_inpaint,
        )
        import fitz
        from PIL import Image as PILImage

        DPI = 200

        # Salva texto nativo ANTES de converter para imagem
        doc_orig = fitz.open(str(pdf_path))
        page_orig = doc_orig[req.page]
        text_dict = page_orig.get_text("dict")
        orig_width  = page_orig.rect.width
        orig_height = page_orig.rect.height
        doc_orig.close()

        # 1. Converte página para imagem
        img = pdf_page_to_image(str(pdf_path), req.page, dpi=DPI)
        ih, iw = img.shape[:2]

        # 2. Converte % → pixels
        x1 = max(0,  int(req.x_pct / 100 * iw))
        y1 = max(0,  int(req.y_pct / 100 * ih))
        x2 = min(iw, int((req.x_pct + req.w_pct) / 100 * iw))
        y2 = min(ih, int((req.y_pct + req.h_pct) / 100 * ih))

        if x2 <= x1 or y2 <= y1:
            raise HTTPException(400, "Área inválida.")

        # 3. Inpainting
        img_result = remove_content_inpaint(
            img, x1, y1, x2, y2,
            full_area=True,
            radius=7,
        )

        # 4. Substitui página no PDF
        doc = fitz.open(str(pdf_path))

        try:
            import cv2
            rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = img_result[:, :, ::-1]

        pil_result = PILImage.fromarray(rgb)
        tmp_page_pdf = str(
            session_path(req.session_id) /
            f"tmp_erase_{req.page}_{uuid.uuid4().hex[:8]}.pdf"
        )
        pil_result.save(tmp_page_pdf, format="PDF", resolution=DPI)

        tmp_doc = fitz.open(tmp_page_pdf)
        doc.delete_page(req.page)
        doc.insert_pdf(tmp_doc, from_page=0, to_page=0, start_at=req.page)
        tmp_doc.close()
        os.remove(tmp_page_pdf)

        # 5. Injeta camada de texto invisível (preserva extração nativa)
        _inject_text_layer(doc, req.page, text_dict, orig_width, orig_height)

        tmp_pdf = str(pdf_path) + ".tmp"
        doc.save(tmp_pdf, garbage=4, deflate=True)
        doc.close()
        os.replace(tmp_pdf, str(pdf_path))

        return {"ok": True, "message": "Área apagada. Fundo reconstruído."}

    except HTTPException:
        raise
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

        # Salva texto nativo ANTES de converter
        doc_orig = fitz.open(str(pdf_path))
        page_orig = doc_orig[page]
        text_dict = page_orig.get_text("dict")
        orig_width  = page_orig.rect.width
        orig_height = page_orig.rect.height
        doc_orig.close()

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

        # Injeta camada de texto invisível
        _inject_text_layer(doc, page, text_dict, orig_width, orig_height)

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
        import fitz

        edits_list = json.loads(edits)
        dpi = 200

        # Salva texto nativo de TODAS as páginas ANTES de converter
        doc_orig = fitz.open(str(pdf_path))
        pages_text = {}
        pages_size = {}
        for i in range(doc_orig.page_count):
            p = doc_orig[i]
            pages_text[i] = p.get_text("dict")
            pages_size[i] = (p.rect.width, p.rect.height)
        doc_orig.close()

        all_imgs = pdf_all_pages_to_images(str(pdf_path), dpi=dpi)

        doc = fitz.open(str(pdf_path))

        by_page = {}
        for e in edits_list:
            by_page.setdefault(e["page"], []).append(e)

        for page_idx, page_edits in by_page.items():
            img = all_imgs[page_idx].copy()
            ih, iw = img.shape[:2]
            pw, ph = pages_size[page_idx]
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

        # Salva PDF com imagens
        tmp = str(pdf_path) + ".tmp"
        image_to_pdf(all_imgs, tmp, dpi=dpi)
        os.replace(tmp, str(pdf_path))

        # Injeta camada de texto invisível em TODAS as páginas
        # — atualiza texto editado nas páginas modificadas
        doc2 = fitz.open(str(pdf_path))

        for page_idx in range(doc2.page_count):
            text_dict = pages_text.get(page_idx)
            if not text_dict:
                continue
            orig_w, orig_h = pages_size[page_idx]

            # Para páginas editadas, atualiza texto dos blocos editados
            page_edits_map = {}
            for edit in by_page.get(page_idx, []):
                page_edits_map[edit["original_text"]] = edit["new_text"]

            _inject_text_layer(
                doc2, page_idx, text_dict, orig_w, orig_h,
                text_replacements=page_edits_map
            )

        tmp2 = str(pdf_path) + ".tmp2"
        doc2.save(tmp2, garbage=4, deflate=True)
        doc2.close()
        os.replace(tmp2, str(pdf_path))

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


# ── Helper: injeta camada de texto invisível ──────────────────────────────

def _inject_text_layer(
    doc: "fitz.Document",
    page_idx: int,
    text_dict: dict,
    orig_width: float,
    orig_height: float,
    text_replacements: dict = None,
):
    """
    Injeta texto invisível (renderMode=3) sobre a página imagem.
    Escala as coordenadas do espaço PDF original para o novo tamanho da página.
    Isso permite que o PyMuPDF extraia texto nativo mesmo após a página
    ter sido convertida para imagem — sem OCR externo.

    text_replacements: dict {texto_original: texto_novo} para páginas editadas
    """
    try:
        import fitz

        page = doc[page_idx]
        new_w = page.rect.width
        new_h = page.rect.height

        # Fatores de escala do espaço original para o novo
        sx = new_w / orig_width  if orig_width  > 0 else 1.0
        sy = new_h / orig_height if orig_height > 0 else 1.0

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    # Aplica substituição se houver
                    if text_replacements and text in text_replacements:
                        text = text_replacements[text]

                    bbox = span["bbox"]
                    x0 = bbox[0] * sx
                    y0 = bbox[1] * sy
                    x1 = bbox[2] * sx
                    y1 = bbox[3] * sy

                    font_size = span.get("size", 11.0) * sy
                    font_size = max(4.0, font_size)

                    # Cor original
                    raw_color = span.get("color", 0)
                    if isinstance(raw_color, int):
                        r = ((raw_color >> 16) & 0xFF) / 255.0
                        g = ((raw_color >> 8) & 0xFF) / 255.0
                        b = (raw_color & 0xFF) / 255.0
                        color = (r, g, b)
                    else:
                        color = tuple(raw_color)

                    # Insere texto invisível (renderMode=3 = invisible)
                    try:
                        page.insert_text(
                            (x0, y1),           # posição baseline
                            text,
                            fontsize=font_size,
                            color=color,
                            render_mode=3,      # invisível mas selecionável/extraível
                        )
                    except Exception as e:
                        print(f"[inject_text] span error: {e}")
                        continue

    except Exception as e:
        print(f"[_inject_text_layer] p{page_idx} error: {e}")
