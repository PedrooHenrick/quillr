"""
backend/main.py
FastAPI — PDF Editor Pro Web
"""
import os
import uuid
import shutil
import base64
import json
import re
import io
import zipfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
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

GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL    = "meta-llama/llama-4-scout-17b-16e-instruct"


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


# ─────────────────────────────────────────────────────────────────────────────
# GROQ OCR — posições em pixels reais
# ─────────────────────────────────────────────────────────────────────────────

def _page_to_base64(pdf_path: str, page_idx: int, dpi: int = 150) -> tuple:
    """
    Renderiza página do PDF como imagem PNG base64.
    Retorna (base64_str, img_width_px, img_height_px).
    """
    import fitz
    from PIL import Image as PILImage

    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    doc.close()

    img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Limita a 1200px no maior lado para economizar tokens
    max_dim = 1200
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), PILImage.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64, img.size[0], img.size[1]


def _groq_ocr(pdf_path: str, page_idx: int, pdf_page_w: float, pdf_page_h: float) -> list:
    """
    Usa Groq Vision para OCR da página imagem.
    Retorna lista de blocos com coordenadas no espaço PDF (pontos).

    Diferencial vs OCR antigo:
    - Groq retorna bbox em PIXELS REAIS da imagem renderizada
    - Convertemos pixels -> coordenadas PDF com escala correta
    - Posições ficam precisas no canvas do frontend
    """
    if not GROQ_API_KEY:
        print("[Groq OCR] GROQ_API_KEY nao configurada.")
        return []

    try:
        from openai import OpenAI
        client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

        b64, img_w, img_h = _page_to_base64(pdf_path, page_idx)

        prompt = f"""Voce e um motor de OCR de alta precisao para documentos PDF.

Esta imagem tem {img_w}x{img_h} pixels.

Extraia TODOS os textos visiveis e retorne as posicoes em PIXELS REAIS desta imagem.

Retorne APENAS um JSON valido sem markdown:
{{
  "blocks": [
    {{
      "text": "texto exato como aparece",
      "x": 120,
      "y": 45,
      "w": 380,
      "h": 22,
      "font_size_px": 18,
      "is_bold": false,
      "is_italic": false,
      "font_family": "serif",
      "align": "left",
      "color_r": 0,
      "color_g": 0,
      "color_b": 0
    }}
  ]
}}

Regras:
- x, y, w, h sao em PIXELS da imagem ({img_w}x{img_h})
- x = borda esquerda do texto
- y = borda superior do texto
- w = largura do bloco de texto
- h = altura do bloco de texto
- font_size_px = altura real dos caracteres em pixels
- Inclua TODOS os textos, inclusive pequenos
- Para texto centralizado use align "center"
- Retorne apenas JSON, sem explicacoes"""

        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ]
            }],
            temperature=0.05,
            max_tokens=4000,
        )

        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            print("[Groq OCR] JSON nao encontrado na resposta")
            return []

        data = json.loads(m.group(0))

        # Fatores de escala: pixels da imagem → pontos PDF
        sx = pdf_page_w / img_w
        sy = pdf_page_h / img_h

        result = []
        for i, b in enumerate(data.get("blocks", [])):
            text = b.get("text", "").strip()
            if not text:
                continue

            # Converte pixels → coordenadas PDF
            x0_pdf = b.get("x", 0) * sx
            y0_pdf = b.get("y", 0) * sy
            x1_pdf = (b.get("x", 0) + b.get("w", 50)) * sx
            y1_pdf = (b.get("y", 0) + b.get("h", 12)) * sy

            # Tamanho da fonte: pixels → pontos PDF
            fs_px  = b.get("font_size_px", b.get("h", 12) * 0.8)
            fs_pdf = max(6.0, fs_px * sy)

            r  = b.get("color_r", 0) / 255.0
            g  = b.get("color_g", 0) / 255.0
            bv = b.get("color_b", 0) / 255.0

            result.append({
                "id": f"p{page_idx}_ocr_{i}",
                "text": text,
                "x0": x0_pdf, "y0": y0_pdf,
                "x1": x1_pdf, "y1": y1_pdf,
                "font_size": fs_pdf,
                "font_name": b.get("font_family", "arial"),
                "is_bold":   b.get("is_bold", False),
                "is_italic": b.get("is_italic", False),
                "color_rgb": [r, g, bv],
                "align":     b.get("align", "left"),
                "source":    "groq_ocr",
            })

        print(f"[Groq OCR] p{page_idx}: {len(result)} blocos extraidos")
        return result

    except Exception as e:
        print(f"[Groq OCR] Erro: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "PDF Editor Pro API online"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Apenas arquivos PDF sao aceitos.")

    session_id = str(uuid.uuid4())
    pdf_path   = get_pdf_path(session_id)

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
        "filename":   file.filename,
        "page_count": page_count,
        "pages":      pages_info,
    }


@app.get("/render/{session_id}/{page}")
async def render_page(session_id: str, page: int, zoom: float = 1.5):
    pdf_path = get_pdf_path(session_id)
    if not pdf_path.exists():
        raise HTTPException(404, "Sessao nao encontrada.")

    import fitz
    doc = fitz.open(str(pdf_path))
    if page < 0 or page >= doc.page_count:
        raise HTTPException(400, "Pagina invalida.")

    pix     = doc[page].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img_path = session_path(session_id) / f"page_{page}_{uuid.uuid4().hex[:8]}.png"
    pix.save(str(img_path))
    doc.close()

    return FileResponse(str(img_path), media_type="image/png")


@app.post("/extract/{session_id}/{page}")
async def extract_text(session_id: str, page: int):
    """
    Extrai texto da pagina.
    - Se tem texto nativo (PyMuPDF): usa direto, rapido e preciso
    - Se pagina virou imagem (pos-edicao): usa Groq OCR com posicoes em pixels reais
    """
    pdf_path = get_pdf_path(session_id)
    if not pdf_path.exists():
        raise HTTPException(404, "Sessao nao encontrada.")

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from core.extractor import PDFExtractor
        import fitz

        extractor = PDFExtractor()
        extractor.open(str(pdf_path))

        mode   = extractor.detect_page_mode(page)
        blocks_native = []

        if mode == "native":
            # Texto nativo disponivel — extrai direto
            blocks_obj    = extractor.extract_page(page)
            blocks_native = [
                {
                    "id":        b.id,
                    "text":      b.text,
                    "x0":        b.x0, "y0": b.y0,
                    "x1":        b.x1, "y1": b.y1,
                    "font_size": b.font_size,
                    "font_name": b.font_name,
                    "is_bold":   b.is_bold,
                    "is_italic": b.is_italic,
                    "color_rgb": list(b.color_rgb),
                    "align":     b.align,
                    "source":    b.source,
                }
                for b in blocks_obj
            ]
            extractor.close()
            return {"blocks": blocks_native}

        # Pagina virou imagem — usa Groq OCR com pixels reais
        extractor.close()

        doc      = fitz.open(str(pdf_path))
        pg       = doc[page]
        pdf_w    = pg.rect.width
        pdf_h    = pg.rect.height
        doc.close()

        print(f"[extract] p{page} modo=imagem -> Groq OCR (pagina {pdf_w:.0f}x{pdf_h:.0f}pt)")
        blocks_ocr = _groq_ocr(str(pdf_path), page, pdf_w, pdf_h)

        if not blocks_ocr:
            return {"blocks": []}

        return {"blocks": blocks_ocr}

    except Exception as e:
        raise HTTPException(500, f"Erro ao extrair texto: {e}")


@app.get("/session-check/{session_id}")
async def session_check(session_id: str):
    pdf_path = get_pdf_path(session_id)
    if not pdf_path.exists():
        raise HTTPException(404, "Sessao nao encontrada.")
    return {"ok": True}


@app.post("/erase")
async def erase_area(req: EraseRequest):
    pdf_path = get_pdf_path(req.session_id)
    if not pdf_path.exists():
        raise HTTPException(404, "Sessao nao encontrada.")

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from core.inpainting_engine import pdf_page_to_image, remove_content_inpaint
        import fitz
        from PIL import Image as PILImage

        DPI = 200

        doc_orig    = fitz.open(str(pdf_path))
        page_orig   = doc_orig[req.page]
        text_dict   = page_orig.get_text("dict")
        orig_width  = page_orig.rect.width
        orig_height = page_orig.rect.height
        doc_orig.close()

        img     = pdf_page_to_image(str(pdf_path), req.page, dpi=DPI)
        ih, iw  = img.shape[:2]

        x1 = max(0,  int(req.x_pct / 100 * iw))
        y1 = max(0,  int(req.y_pct / 100 * ih))
        x2 = min(iw, int((req.x_pct + req.w_pct) / 100 * iw))
        y2 = min(ih, int((req.y_pct + req.h_pct) / 100 * ih))

        if x2 <= x1 or y2 <= y1:
            raise HTTPException(400, "Area invalida.")

        img_result = remove_content_inpaint(img, x1, y1, x2, y2, full_area=True, radius=7)

        doc = fitz.open(str(pdf_path))

        try:
            import cv2
            rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = img_result[:, :, ::-1]

        pil_result   = PILImage.fromarray(rgb)
        tmp_page_pdf = str(session_path(req.session_id) / f"tmp_erase_{req.page}_{uuid.uuid4().hex[:8]}.pdf")
        pil_result.save(tmp_page_pdf, format="PDF", resolution=DPI)

        tmp_doc = fitz.open(tmp_page_pdf)
        doc.delete_page(req.page)
        doc.insert_pdf(tmp_doc, from_page=0, to_page=0, start_at=req.page)
        tmp_doc.close()
        os.remove(tmp_page_pdf)

        _inject_text_layer(doc, req.page, text_dict, orig_width, orig_height)

        tmp_pdf = str(pdf_path) + ".tmp"
        doc.save(tmp_pdf, garbage=4, deflate=True)
        doc.close()
        os.replace(tmp_pdf, str(pdf_path))

        return {"ok": True, "message": "Area apagada. Fundo reconstruido."}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Erro ao apagar: {e}")


@app.post("/signature")
async def add_signature(
    session_id: str = Form(...),
    page:       int   = Form(...),
    x_pct:      float = Form(...),
    y_pct:      float = Form(...),
    w_pct:      float = Form(...),
    h_pct:      float = Form(...),
    file:       UploadFile = File(...),
):
    pdf_path = get_pdf_path(session_id)
    if not pdf_path.exists():
        raise HTTPException(404, "Sessao nao encontrada.")

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from core.inpainting_engine import pdf_page_to_image
        import fitz
        import numpy as np
        from PIL import Image as PILImage

        doc_orig    = fitz.open(str(pdf_path))
        page_orig   = doc_orig[page]
        text_dict   = page_orig.get_text("dict")
        orig_width  = page_orig.rect.width
        orig_height = page_orig.rect.height
        doc_orig.close()

        sig_path = session_path(session_id) / f"sig_{uuid.uuid4()}.png"
        with open(sig_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        dpi     = 200
        img     = pdf_page_to_image(str(pdf_path), page, dpi=dpi)
        ih, iw  = img.shape[:2]

        x1 = max(0,  int(x_pct / 100 * iw))
        y1 = max(0,  int(y_pct / 100 * ih))
        w  = max(10, int(w_pct / 100 * iw))
        h  = max(10, int(h_pct / 100 * ih))

        sig  = PILImage.open(str(sig_path)).convert("RGBA")
        sig  = sig.resize((w, h), PILImage.LANCZOS)
        base = PILImage.fromarray(img[:, :, ::-1])
        base.paste(sig, (x1, y1), sig.split()[3])

        img_result = np.array(base)[:, :, ::-1]

        try:
            import cv2
            rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = img_result[:, :, ::-1]

        pil_result = PILImage.fromarray(rgb)
        doc        = fitz.open(str(pdf_path))
        tmp_page   = str(session_path(session_id) / f"tmp_sig_{page}.pdf")
        pil_result.save(tmp_page, format="PDF", resolution=dpi)

        tmp_doc = fitz.open(tmp_page)
        doc.delete_page(page)
        doc.insert_pdf(tmp_doc, from_page=0, to_page=0, start_at=page)
        tmp_doc.close()
        os.remove(tmp_page)
        os.remove(str(sig_path))

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
    edits:      str = Form(...),
):
    pdf_path = get_pdf_path(session_id)
    if not pdf_path.exists():
        raise HTTPException(404, "Sessao nao encontrada.")

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from core.inpainting_engine import (
            pdf_all_pages_to_images, remove_content_inpaint,
            smart_replace_text, image_to_pdf)
        import fitz

        edits_list = json.loads(edits)
        dpi        = 200

        doc_orig   = fitz.open(str(pdf_path))
        pages_text = {}
        pages_size = {}
        for i in range(doc_orig.page_count):
            p = doc_orig[i]
            pages_text[i] = p.get_text("dict")
            pages_size[i] = (p.rect.width, p.rect.height)
        doc_orig.close()

        all_imgs = pdf_all_pages_to_images(str(pdf_path), dpi=dpi)
        doc      = fitz.open(str(pdf_path))

        by_page = {}
        for e in edits_list:
            by_page.setdefault(e["page"], []).append(e)

        for page_idx, page_edits in by_page.items():
            img    = all_imgs[page_idx].copy()
            ih, iw = img.shape[:2]
            pw, ph = pages_size[page_idx]
            sx, sy = iw / pw, ih / ph

            for edit in page_edits:
                x1 = max(0,  int(edit["x0"] * sx))
                y1 = max(0,  int(edit["y0"] * sy))
                x2 = min(iw, int(edit["x1"] * sx))
                y2 = min(ih, int(edit["y1"] * sy))
                img = remove_content_inpaint(img, x1, y1, x2, y2, threshold=80, full_area=False, radius=5)
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

        doc2 = fitz.open(str(pdf_path))
        for page_idx in range(doc2.page_count):
            text_dict = pages_text.get(page_idx)
            if not text_dict:
                continue
            orig_w, orig_h    = pages_size[page_idx]
            page_edits_map    = {e["original_text"]: e["new_text"] for e in by_page.get(page_idx, [])}
            _inject_text_layer(doc2, page_idx, text_dict, orig_w, orig_h, text_replacements=page_edits_map)

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
        raise HTTPException(404, "Sessao nao encontrada.")
    return FileResponse(str(pdf_path), media_type="application/pdf", filename="documento_editado.pdf")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    p = session_path(session_id)
    if p.exists():
        shutil.rmtree(str(p))
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────────
# DIVIDIR PDF
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/split/info")
async def split_info(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Apenas arquivos PDF sao aceitos.")
    try:
        import fitz
        contents   = await file.read()
        doc        = fitz.open(stream=contents, filetype="pdf")
        page_count = doc.page_count
        doc.close()
        return {"page_count": page_count, "filename": file.filename}
    except Exception as e:
        raise HTTPException(500, f"Erro ao ler PDF: {e}")


@app.post("/split/pages")
async def split_pages(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Apenas arquivos PDF sao aceitos.")
    try:
        import fitz
        contents  = await file.read()
        doc       = fitz.open(stream=contents, filetype="pdf")
        base_name = file.filename.replace(".pdf", "")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(doc.page_count):
                writer    = fitz.open()
                writer.insert_pdf(doc, from_page=i, to_page=i)
                pdf_bytes = writer.tobytes(garbage=4, deflate=True)
                writer.close()
                zf.writestr(f"{base_name}_pagina_{i + 1}.pdf", pdf_bytes)

        doc.close()
        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{base_name}_dividido.zip"'},
        )
    except Exception as e:
        raise HTTPException(500, f"Erro ao dividir PDF: {e}")


@app.post("/split/range")
async def split_range(
    file:   UploadFile = File(...),
    ranges: str        = Form(...),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Apenas arquivos PDF sao aceitos.")
    try:
        import fitz
        ranges_list = json.loads(ranges)
        contents    = await file.read()
        doc         = fitz.open(stream=contents, filetype="pdf")
        base_name   = file.filename.replace(".pdf", "")
        total       = doc.page_count

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, r in enumerate(ranges_list):
                from_page = max(0, int(r["from"]))
                to_page   = min(total - 1, int(r["to"]))
                if from_page > to_page:
                    continue
                writer    = fitz.open()
                writer.insert_pdf(doc, from_page=from_page, to_page=to_page)
                pdf_bytes = writer.tobytes(garbage=4, deflate=True)
                writer.close()
                label = f"p{from_page + 1}-{to_page + 1}"
                zf.writestr(f"{base_name}_parte_{i + 1}_{label}.pdf", pdf_bytes)

        doc.close()
        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{base_name}_dividido.zip"'},
        )
    except Exception as e:
        raise HTTPException(500, f"Erro ao dividir PDF por intervalo: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: camada de texto invisivel
# ─────────────────────────────────────────────────────────────────────────────

def _inject_text_layer(
    doc: "fitz.Document",
    page_idx: int,
    text_dict: dict,
    orig_width: float,
    orig_height: float,
    text_replacements: dict = None,
):
    """
    Injeta texto invisivel sobre a pagina imagem.
    Escala coordenadas do espaco PDF original para o novo tamanho.
    Permite extracao nativa sem OCR nas proximas sessoes.
    """
    try:
        import fitz

        page  = doc[page_idx]
        new_w = page.rect.width
        new_h = page.rect.height
        sx    = new_w / orig_width  if orig_width  > 0 else 1.0
        sy    = new_h / orig_height if orig_height > 0 else 1.0

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    if text_replacements and text in text_replacements:
                        text = text_replacements[text]

                    bbox      = span["bbox"]
                    x0        = bbox[0] * sx
                    y1_base   = bbox[3] * sy
                    font_size = max(4.0, span.get("size", 11.0) * sy)

                    try:
                        page.insert_text(
                            (x0, y1_base),
                            text,
                            fontsize=font_size,
                            color=(0, 0, 0),
                            render_mode=3,
                        )
                    except Exception as e2:
                        print(f"[inject_text] erro: {e2}")

    except Exception as e:
        print(f"[_inject_text_layer] p{page_idx} erro: {e}")
