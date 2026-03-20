"""
backend/main.py
FastAPI — Quillr PDF API
Armazenamento temporário: Supabase Storage (PDFs deletados ao fim da sessão)
Segurança: JWT Supabase, Rate Limiting, CORS restrito, validação de upload,
           sessão vinculada ao usuário, logs de segurança
"""
import os
import sys
import uuid
import tempfile
import shutil
import time
import json
import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict
from datetime import datetime

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

# ── Logs de segurança ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("quillr.security")

def log_event(event: str, user_id: str = "-", ip: str = "-", details: str = ""):
    logger.info(f"EVENT={event} USER={user_id} IP={ip} {details}")

def log_warning(event: str, user_id: str = "-", ip: str = "-", details: str = ""):
    logger.warning(f"[AVISO] EVENT={event} USER={user_id} IP={ip} {details}")

def log_error(event: str, user_id: str = "-", ip: str = "-", details: str = ""):
    logger.error(f"[ERRO] EVENT={event} USER={user_id} IP={ip} {details}")

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Quillr PDF API")

IS_PRODUCTION = os.environ.get("ENVIRONMENT", "development") == "production"
if IS_PRODUCTION:
    logger.info("Ambiente de producao detectado (Railway)")
else:
    logger.info("Ambiente de desenvolvimento")

# ── Headers de segurança HTTP ──────────────────────────────────────────────
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    if IS_PRODUCTION:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# ── CORS ───────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:5174"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# ── Rate Limiting ──────────────────────────────────────────────────────────
RATE_STORE: dict[str, list] = defaultdict(list)

RATE_LIMITS = {
    "upload":    (5,  60),
    "extract":   (20, 60),
    "erase":     (10, 60),
    "signature": (10, 60),
    "save_text": (10, 60),
    "default":   (60, 60),
}

def check_rate_limit(request: Request, endpoint: str = "default", user_id: str = "-"):
    ip = request.client.host
    limit, window = RATE_LIMITS.get(endpoint, RATE_LIMITS["default"])
    now = time.time()
    RATE_STORE[ip] = [(t, e) for t, e in RATE_STORE[ip] if now - t < window]
    endpoint_count = sum(1 for _, e in RATE_STORE[ip] if e == endpoint)
    if endpoint_count >= limit:
        log_warning("RATE_LIMIT_HIT", user_id=user_id, ip=ip,
                    details=f"endpoint={endpoint} count={endpoint_count} limit={limit}")
        raise HTTPException(status_code=429,
                            detail=f"Muitas requisições. Tente novamente em {window} segundos.")
    RATE_STORE[ip].append((now, endpoint))

# ── Autenticação JWT Supabase ──────────────────────────────────────────────
security = HTTPBearer(auto_error=False)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "pdf-sessions")

_TOKEN_CACHE: dict = {}

async def verify_token(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    tk: str = "",
) -> dict:
    if credentials and credentials.credentials:
        token = credentials.credentials
    elif tk:
        token = tk
    else:
        raise HTTPException(401, "Token nao fornecido.")
    ip = request.client.host

    import hashlib
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    now = time.time()

    if token_hash in _TOKEN_CACHE:
        cached_user, expiry = _TOKEN_CACHE[token_hash]
        if now < expiry:
            return cached_user
        else:
            del _TOKEN_CACHE[token_hash]

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        log_error("SUPABASE_NOT_CONFIGURED", ip=ip)
        raise HTTPException(500, "Supabase nao configurado no servidor.")

    try:
        import httpx
        response = httpx.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": SUPABASE_SERVICE_KEY,
            },
            timeout=5.0,
        )
        if response.status_code != 200:
            log_warning("INVALID_TOKEN", ip=ip, details=f"status={response.status_code}")
            raise HTTPException(401, "Token invalido ou expirado. Faca login novamente.")

        user_data = response.json()
        if not user_data.get("id"):
            log_warning("TOKEN_NO_USER", ip=ip)
            raise HTTPException(401, "Usuario nao encontrado.")

        _TOKEN_CACHE[token_hash] = (user_data, now + 10 * 60)
        return user_data

    except HTTPException:
        raise
    except Exception as e:
        log_error("TOKEN_VERIFY_ERROR", ip=ip, details=str(e))
        raise HTTPException(401, "Erro ao validar token.")

# ── Sessão vinculada ao usuário ────────────────────────────────────────────
SESSION_OWNERS: dict[str, str] = {}

def register_session(session_id: str, user_id: str):
    SESSION_OWNERS[session_id] = user_id
    log_event("SESSION_CREATED", user_id=user_id, details=f"session={session_id}")

def verify_session_owner(session_id: str, user_id: str, ip: str = "-"):
    owner = SESSION_OWNERS.get(session_id)
    if owner is not None and owner != user_id:
        log_error("SESSION_UNAUTHORIZED", user_id=user_id, ip=ip,
                  details=f"session={session_id} owner={owner}")
        raise HTTPException(403, "Acesso negado. Esta sessão pertence a outro usuário.")

# ── Supabase Storage helpers ───────────────────────────────────────────────
def storage_path(user_id: str, session_id: str) -> str:
    """Caminho no bucket: user_id/session_id/document.pdf"""
    return f"{user_id}/{session_id}/document.pdf"

def upload_to_storage(user_id: str, session_id: str, content: bytes) -> bool:
    """Faz upload do PDF para o Supabase Storage."""
    import httpx
    path = storage_path(user_id, session_id)
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "apikey": SUPABASE_SERVICE_KEY,
        "Content-Type": "application/pdf",
        "x-upsert": "true",
    }
    res = httpx.post(url, content=content, headers=headers, timeout=60.0)
    return res.status_code in (200, 201)

def download_from_storage(user_id: str, session_id: str) -> bytes:
    """Baixa o PDF do Supabase Storage."""
    import httpx
    path = storage_path(user_id, session_id)
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "apikey": SUPABASE_SERVICE_KEY,
    }
    res = httpx.get(url, headers=headers, timeout=60.0)
    if res.status_code != 200:
        raise HTTPException(404, "Sessão não encontrada.")
    return res.content

def delete_from_storage(user_id: str, session_id: str):
    """Deleta o PDF do Supabase Storage."""
    import httpx
    path = storage_path(user_id, session_id)
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "apikey": SUPABASE_SERVICE_KEY,
    }
    httpx.delete(url, headers=headers, timeout=10.0)

def get_pdf_local(user_id: str, session_id: str) -> Path:
    """
    Baixa o PDF do Supabase Storage para um arquivo temporário local.
    Retorna o path local temporário.
    """
    content = download_from_storage(user_id, session_id)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"quillr_{session_id}_"))
    pdf_path = tmp_dir / "document.pdf"
    pdf_path.write_bytes(content)
    return pdf_path

def save_pdf_and_cleanup(user_id: str, session_id: str, pdf_path: Path):
    """
    Faz upload do PDF modificado de volta para o Supabase Storage
    e remove o diretório temporário local.
    """
    content = pdf_path.read_bytes()
    upload_to_storage(user_id, session_id, content)
    shutil.rmtree(str(pdf_path.parent), ignore_errors=True)

# ── Validação de PDF ───────────────────────────────────────────────────────
MAX_FILE_SIZE = 100 * 1024 * 1024
PDF_MAGIC_BYTES = b"%PDF"

async def validate_pdf(file: UploadFile, user_id: str = "-") -> bytes:
    if not file.filename.lower().endswith(".pdf"):
        log_warning("INVALID_FILE_EXT", user_id=user_id, details=f"file={file.filename}")
        raise HTTPException(400, "Apenas arquivos PDF são aceitos.")

    content = await file.read()
    await file.seek(0)

    if len(content) > MAX_FILE_SIZE:
        log_warning("FILE_TOO_LARGE", user_id=user_id,
                    details=f"size={len(content)} max={MAX_FILE_SIZE}")
        raise HTTPException(413, "Arquivo muito grande. Máximo: 100MB.")

    if len(content) < 4 or not content.startswith(PDF_MAGIC_BYTES):
        log_warning("INVALID_PDF_MAGIC", user_id=user_id, details=f"file={file.filename}")
        raise HTTPException(400, "Arquivo não é um PDF válido.")

    if b"/JavaScript" in content or b"/JS " in content:
        log_error("PDF_WITH_JAVASCRIPT", user_id=user_id, details=f"file={file.filename}")
        raise HTTPException(400, "PDF contém JavaScript e não é permitido.")

    try:
        import magic as magic_lib
        mime = magic_lib.from_buffer(content[:2048], mime=True)
        if mime != "application/pdf":
            log_warning("INVALID_MIME", user_id=user_id,
                        details=f"mime={mime} file={file.filename}")
            raise HTTPException(400, f"Tipo de arquivo não permitido: {mime}")
    except ImportError:
        pass

    return content

# ── Models ─────────────────────────────────────────────────────────────────
class EraseRequest(BaseModel):
    session_id: str
    page: int
    x_pct: float
    y_pct: float
    w_pct: float
    h_pct: float

# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Quillr PDF API online"}

@app.get("/session-check/{session_id}")
async def session_check(
    session_id: str,
    request: Request,
    user: dict = Depends(verify_token),
):
    user_id = user["id"]
    try:
        content = download_from_storage(user_id, session_id)
    except HTTPException:
        return {"valid": False}

    register_session(session_id, user_id)

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"quillr_{session_id}_"))
    pdf_path = tmp_dir / "document.pdf"
    pdf_path.write_bytes(content)

    import fitz
    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    pages_info = [{"width": doc[i].rect.width, "height": doc[i].rect.height}
                  for i in range(page_count)]
    doc.close()
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    log_event("SESSION_REUSED", user_id=user_id, ip=request.client.host,
              details=f"session={session_id}")

    return {
        "valid": True,
        "session_id": session_id,
        "page_count": page_count,
        "pages": pages_info,
        "filename": "documento_editado.pdf",
    }

@app.post("/upload")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(verify_token),
):
    user_id = user["id"]
    ip = request.client.host
    check_rate_limit(request, "upload", user_id)

    content = await validate_pdf(file, user_id)
    session_id = str(uuid.uuid4())

    # Salva no Supabase Storage
    ok = upload_to_storage(user_id, session_id, content)
    if not ok:
        log_error("STORAGE_UPLOAD_FAILED", user_id=user_id, ip=ip)
        raise HTTPException(500, "Erro ao salvar arquivo no storage.")

    register_session(session_id, user_id)

    # Abre localmente só para ler metadados
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"quillr_{session_id}_"))
    pdf_path = tmp_dir / "document.pdf"
    pdf_path.write_bytes(content)

    import fitz
    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    pages_info = [{"width": doc[i].rect.width, "height": doc[i].rect.height}
                  for i in range(page_count)]
    doc.close()
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    log_event("UPLOAD_SUCCESS", user_id=user_id, ip=ip,
              details=f"file={file.filename} pages={page_count} session={session_id}")

    return {
        "session_id": session_id,
        "filename": file.filename,
        "page_count": page_count,
        "pages": pages_info,
    }

@app.get("/render/{session_id}/{page}")
async def render_page(
    session_id: str,
    page: int,
    request: Request,
    zoom: float = 1.5,
    tk: str = "",
    user: dict = Depends(verify_token),
):
    user_id = user["id"]
    ip = request.client.host
    check_rate_limit(request, "default", user_id)
    verify_session_owner(session_id, user_id, ip)

    zoom = max(0.5, min(zoom, 3.0))
    pdf_path = get_pdf_local(user_id, session_id)

    import fitz
    doc = fitz.open(str(pdf_path))
    if page < 0 or page >= doc.page_count:
        doc.close()
        shutil.rmtree(str(pdf_path.parent), ignore_errors=True)
        raise HTTPException(400, "Página inválida.")

    pix = doc[page].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img_bytes = pix.tobytes("png")
    doc.close()
    shutil.rmtree(str(pdf_path.parent), ignore_errors=True)

    import io
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")

@app.post("/extract/{session_id}/{page}")
async def extract_text(
    session_id: str,
    page: int,
    request: Request,
    user: dict = Depends(verify_token),
):
    user_id = user["id"]
    ip = request.client.host
    check_rate_limit(request, "extract", user_id)
    verify_session_owner(session_id, user_id, ip)

    pdf_path = get_pdf_local(user_id, session_id)

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from core.extractor import PDFExtractor

        groq_key = os.environ.get("GROQ_API_KEY", "")
        extractor = PDFExtractor(groq_api_key=groq_key)
        extractor.open(str(pdf_path))
        blocks = extractor.extract_page(page)

        result = [
            {
                "id": b.id, "text": b.text,
                "x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1,
                "font_size": b.font_size, "font_name": b.font_name,
                "is_bold": b.is_bold, "is_italic": b.is_italic,
                "color_rgb": list(b.color_rgb), "align": b.align,
                "source": b.source,
            }
            for b in blocks
        ]
        extractor.close()
    finally:
        shutil.rmtree(str(pdf_path.parent), ignore_errors=True)

    log_event("EXTRACT_SUCCESS", user_id=user_id, ip=ip,
              details=f"session={session_id} page={page} blocks={len(result)}")
    return {"blocks": result}

@app.post("/erase")
async def erase_area(
    req: EraseRequest,
    request: Request,
    user: dict = Depends(verify_token),
):
    user_id = user["id"]
    ip = request.client.host
    check_rate_limit(request, "erase", user_id)
    verify_session_owner(req.session_id, user_id, ip)

    for val in [req.x_pct, req.y_pct, req.w_pct, req.h_pct]:
        if not (0 <= val <= 100):
            raise HTTPException(400, "Coordenadas inválidas.")

    pdf_path = get_pdf_local(user_id, req.session_id)

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from core.inpainting_engine import pdf_page_to_image, remove_content_inpaint
        import fitz
        from PIL import Image as PILImage

        dpi = 200
        img = pdf_page_to_image(str(pdf_path), req.page, dpi=dpi)
        ih, iw = img.shape[:2]

        x1 = max(0,  int(req.x_pct / 100 * iw))
        y1 = max(0,  int(req.y_pct / 100 * ih))
        x2 = min(iw, int((req.x_pct + req.w_pct) / 100 * iw))
        y2 = min(ih, int((req.y_pct + req.h_pct) / 100 * ih))

        img_clean = remove_content_inpaint(img, x1, y1, x2, y2,
                                           threshold=80, full_area=True, radius=7)

        try:
            import cv2
            rgb = cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB)
        except Exception:
            rgb = img_clean[:, :, ::-1]
        pil_clean = PILImage.fromarray(rgb)

        doc = fitz.open(str(pdf_path))
        tmp_page = str(pdf_path.parent / f"tmp_page_{req.page}.pdf")
        pil_clean.save(tmp_page, format="PDF", resolution=dpi)
        tmp_doc = fitz.open(tmp_page)
        doc.delete_page(req.page)
        doc.insert_pdf(tmp_doc, from_page=0, to_page=0, start_at=req.page)
        tmp_doc.close()
        os.remove(tmp_page)

        tmp_pdf = str(pdf_path) + ".tmp"
        doc.save(tmp_pdf, garbage=4, deflate=True)
        doc.close()
        os.replace(tmp_pdf, str(pdf_path))

        save_pdf_and_cleanup(user_id, req.session_id, pdf_path)

        log_event("ERASE_SUCCESS", user_id=user_id, ip=ip,
                  details=f"session={req.session_id} page={req.page}")
        return {"ok": True, "message": "Área apagada com sucesso.", "page_is_image": True}

    except HTTPException:
        shutil.rmtree(str(pdf_path.parent), ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(str(pdf_path.parent), ignore_errors=True)
        log_error("ERASE_ERROR", user_id=user_id, ip=ip, details=str(e))
        raise HTTPException(500, f"Erro ao apagar: {e}")

@app.post("/signature")
async def add_signature(
    request: Request,
    session_id: str = Form(...),
    page: int = Form(...),
    x_pct: float = Form(...),
    y_pct: float = Form(...),
    w_pct: float = Form(...),
    h_pct: float = Form(...),
    file: UploadFile = File(...),
    user: dict = Depends(verify_token),
):
    user_id = user["id"]
    ip = request.client.host
    check_rate_limit(request, "signature", user_id)
    verify_session_owner(session_id, user_id, ip)

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Apenas imagens são aceitas para assinatura.")

    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(413, "Imagem muito grande. Máximo: 5MB.")

    pdf_path = get_pdf_local(user_id, session_id)

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from core.inpainting_engine import pdf_page_to_image
        import fitz
        import numpy as np
        from PIL import Image as PILImage

        sig_path = pdf_path.parent / f"sig_{uuid.uuid4()}.png"
        sig_path.write_bytes(content)

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
        tmp_page = str(pdf_path.parent / f"tmp_sig_{page}.pdf")
        pil_result.save(tmp_page, format="PDF", resolution=dpi)
        tmp_doc = fitz.open(tmp_page)
        doc.delete_page(page)
        doc.insert_pdf(tmp_doc, from_page=0, to_page=0, start_at=page)
        tmp_doc.close()
        os.remove(tmp_page)

        tmp_pdf = str(pdf_path) + ".tmp"
        doc.save(tmp_pdf, garbage=4, deflate=True)
        doc.close()
        os.replace(tmp_pdf, str(pdf_path))

        save_pdf_and_cleanup(user_id, session_id, pdf_path)

        log_event("SIGNATURE_SUCCESS", user_id=user_id, ip=ip,
                  details=f"session={session_id} page={page}")
        return {"ok": True}

    except HTTPException:
        shutil.rmtree(str(pdf_path.parent), ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(str(pdf_path.parent), ignore_errors=True)
        log_error("SIGNATURE_ERROR", user_id=user_id, ip=ip, details=str(e))
        raise HTTPException(500, f"Erro ao inserir assinatura: {e}")

@app.post("/save-text")
async def save_text_edits(
    request: Request,
    session_id: str = Form(...),
    edits: str = Form(...),
    user: dict = Depends(verify_token),
):
    user_id = user["id"]
    ip = request.client.host
    check_rate_limit(request, "save_text", user_id)
    verify_session_owner(session_id, user_id, ip)

    import json as json_lib
    pdf_path = get_pdf_local(user_id, session_id)

    try:
        edits_list = json_lib.loads(edits)
        if len(edits_list) > 500:
            raise HTTPException(400, "Muitas edições por vez. Máximo: 500.")

        sys.path.insert(0, str(Path(__file__).parent))
        from core.inpainting_engine import save_text_edits_native

        tmp = str(pdf_path) + ".tmp"

        # Substitui texto diretamente no PDF vetorial (não converte para imagem)
        # O PDF continua editável após salvar
        save_text_edits_native(str(pdf_path), edits_list, tmp)
        os.replace(tmp, str(pdf_path))

        save_pdf_and_cleanup(user_id, session_id, pdf_path)

        log_event("SAVE_TEXT_SUCCESS", user_id=user_id, ip=ip,
                  details=f"session={session_id} edits={len(edits_list)}")
        return {"ok": True}

    except HTTPException:
        shutil.rmtree(str(pdf_path.parent), ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(str(pdf_path.parent), ignore_errors=True)
        log_error("SAVE_TEXT_ERROR", user_id=user_id, ip=ip, details=str(e))
        raise HTTPException(500, f"Erro ao salvar texto: {e}")

@app.get("/download/{session_id}")
async def download_pdf(
    session_id: str,
    request: Request,
    tk: str = "",
    user: dict = Depends(verify_token),
):
    user_id = user["id"]
    ip = request.client.host
    check_rate_limit(request, "default", user_id)
    verify_session_owner(session_id, user_id, ip)

    content = download_from_storage(user_id, session_id)

    log_event("DOWNLOAD", user_id=user_id, ip=ip, details=f"session={session_id}")

    import io
    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=documento_editado.pdf"}
    )

# ── Mercado Pago ───────────────────────────────────────────────────────────
MP_ACCESS_TOKEN = os.environ.get("MP_ACCESS_TOKEN", "")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")

@app.post("/create-subscription")
async def create_subscription(
    request: Request,
    user: dict = Depends(verify_token),
):
    user_id = user["id"]
    ip = request.client.host

    if not MP_ACCESS_TOKEN:
        raise HTTPException(500, "Mercado Pago nao configurado.")

    body = await request.json()
    email = body.get("email", "")

    try:
        import httpx
        headers = {
            "Authorization": f"Bearer {MP_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        }

        back_url = FRONTEND_URL + "/payment/success"
        is_local = "localhost" in FRONTEND_URL or "127.0.0.1" in FRONTEND_URL

        payload = {
            "reason": "Quillr Pro - Assinatura Mensal",
            "auto_recurring": {
                "frequency": 1,
                "frequency_type": "months",
                "transaction_amount": 29.00,
                "currency_id": "BRL",
            },
            "payer_email": email,
            "external_reference": user_id,
        }

        if not is_local:
            payload["back_url"] = back_url
            payload["notification_url"] = FRONTEND_URL.rstrip("/") + "/webhook/mp"

        response = httpx.post(
            "https://api.mercadopago.com/preapproval",
            headers=headers,
            json=payload,
            timeout=10.0,
        )

        if response.status_code not in (200, 201):
            log_error("MP_CREATE_SUB_ERROR", user_id=user_id, ip=ip,
                      details=response.text)
            raise HTTPException(500, "Erro ao criar assinatura no Mercado Pago.")

        data = response.json()
        log_event("MP_SUB_CREATED", user_id=user_id, ip=ip,
                  details=f"preapproval_id={data.get('id')}")

        return {"init_point": data.get("init_point"), "id": data.get("id")}

    except HTTPException:
        raise
    except Exception as e:
        log_error("MP_CREATE_SUB_EXCEPTION", user_id=user_id, ip=ip, details=str(e))
        raise HTTPException(500, f"Erro: {e}")

@app.post("/webhook/mp")
async def webhook_mercadopago(request: Request):
    try:
        body = await request.json()
        topic = body.get("type") or request.query_params.get("topic", "")
        resource_id = body.get("data", {}).get("id") or request.query_params.get("id", "")

        log_event("MP_WEBHOOK", details=f"topic={topic} id={resource_id}")

        if not MP_ACCESS_TOKEN or not resource_id:
            return {"ok": True}

        import httpx
        headers = {"Authorization": f"Bearer {MP_ACCESS_TOKEN}"}

        if topic in ("subscription_preapproval", "preapproval"):
            res = httpx.get(
                f"https://api.mercadopago.com/preapproval/{resource_id}",
                headers=headers, timeout=10.0
            )
            if res.status_code != 200:
                return {"ok": True}

            sub = res.json()
            user_id = sub.get("external_reference")
            status = sub.get("status")

            if not user_id:
                return {"ok": True}

            new_plan = "pro" if status == "authorized" else "free"

            httpx.patch(
                f"{SUPABASE_URL}/rest/v1/profiles?user_id=eq.{user_id}",
                headers={
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
                json={
                    "plan": new_plan,
                    "mp_subscription_id": resource_id,
                    "mp_subscription_status": status,
                },
                timeout=10.0,
            )

            log_event("MP_PLAN_UPDATED", user_id=user_id,
                      details=f"plan={new_plan} status={status}")

        return {"ok": True}

    except Exception as e:
        log_error("WEBHOOK_ERROR", details=str(e))
        return {"ok": True}

@app.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    request: Request,
    user: dict = Depends(verify_token),
):
    user_id = user["id"]
    ip = request.client.host
    verify_session_owner(session_id, user_id, ip)

    delete_from_storage(user_id, session_id)
    SESSION_OWNERS.pop(session_id, None)

    log_event("SESSION_DELETED", user_id=user_id, ip=ip,
              details=f"session={session_id}")
    return {"ok": True}
