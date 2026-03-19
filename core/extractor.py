"""
core/extractor.py
═══════════════════════════════════════════════════════════════════
Motor de extração inteligente de texto do PDF.

Estratégia:
  1. PyMuPDF extrai todos os blocos de texto nativos (posição, fonte, tamanho, cor)
  2. Groq Vision confirma/corrige fonte e estilo visualmente
  3. Se PDF for escaneado (sem texto nativo), usa Groq OCR na imagem inteira
  4. Retorna lista de TextBlock com tudo que o editor precisa
═══════════════════════════════════════════════════════════════════
"""

import fitz
import json
import base64
import io
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextBlock:
    """Um bloco de texto extraído do PDF com todas as propriedades."""
    id: str                          # identificador único ex: "p0_b3_s1"
    page: int                        # índice da página (0-based)
    text: str                        # texto original
    x0: float                        # bbox no espaço PDF (pontos)
    y0: float
    x1: float
    y1: float
    font_size: float = 12.0          # tamanho em pontos PDF
    font_name: str = "helv"          # nome da fonte PyMuPDF
    font_family: str = "sans-serif"  # serif / sans-serif / monospace
    is_bold: bool = False
    is_italic: bool = False
    color_rgb: tuple = (0.0, 0.0, 0.0)
    align: str = "left"
    source: str = "pymupdf"          # "pymupdf" | "groq_ocr" | "groq_vision"
    # Texto editado pelo usuário (None = não editado)
    edited_text: Optional[str] = None

    @property
    def display_text(self) -> str:
        return self.edited_text if self.edited_text is not None else self.text

    @property
    def is_edited(self) -> bool:
        return self.edited_text is not None and self.edited_text != self.text

    @property
    def bbox(self) -> tuple:
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


class PDFExtractor:
    """
    Extrai todos os blocos de texto de um PDF com propriedades completas.
    Combina PyMuPDF (estrutura) + Groq Vision (confirmação visual).
    """

    def __init__(self, groq_api_key: str = ""):
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self._blocks_cache: dict[int, list[TextBlock]] = {}  # page → blocks
        self.doc: Optional[fitz.Document] = None
        self.file_path: str = ""
        self.page_count: int = 0
        # Modo: "native" (tem texto) ou "scanned" (imagem)
        self.page_modes: dict[int, str] = {}

    def open(self, path: str) -> bool:
        try:
            if self.doc:
                self.doc.close()
            self.doc = fitz.open(path)
            self.file_path = path
            self.page_count = self.doc.page_count
            self._blocks_cache.clear()
            self.page_modes.clear()
            return True
        except Exception as e:
            print(f"[Extractor] open error: {e}")
            return False

    def close(self):
        if self.doc:
            self.doc.close()
            self.doc = None

    # ── Detecção de modo ───────────────────────────────────────────────────

    def detect_page_mode(self, page_idx: int) -> str:
        """Detecta se a página tem texto nativo ou é escaneada."""
        if page_idx in self.page_modes:
            return self.page_modes[page_idx]
        if not self.doc:
            return "unknown"
        page = self.doc[page_idx]
        # Conta spans de texto nativos (mais confiável que contar chars)
        blocks = page.get_text("dict")["blocks"]
        span_count = sum(
            len(line.get("spans", []))
            for b in blocks if b.get("type") == 0
            for line in b.get("lines", [])
        )
        # Se tem pelo menos 1 span nativo = PDF com texto
        mode = "native" if span_count > 0 else "scanned"
        self.page_modes[page_idx] = mode
        return mode

    # ── Extração PyMuPDF ───────────────────────────────────────────────────

    def extract_native(self, page_idx: int) -> list[TextBlock]:
        """Extrai texto nativo via PyMuPDF — rápido e preciso em posição."""
        if not self.doc:
            return []
        page = self.doc[page_idx]
        blocks_raw = page.get_text("dict")["blocks"]
        result = []
        span_counter = 0

        for bi, block in enumerate(blocks_raw):
            if block.get("type") != 0:  # só texto
                continue
            for li, line in enumerate(block.get("lines", [])):
                for si, span in enumerate(line.get("spans", [])):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    span_counter += 1
                    bid = f"p{page_idx}_b{bi}_l{li}_s{si}"

                    # Cor
                    raw_color = span.get("color", 0)
                    if isinstance(raw_color, int):
                        r = ((raw_color >> 16) & 0xFF) / 255.0
                        g = ((raw_color >> 8) & 0xFF) / 255.0
                        b = (raw_color & 0xFF) / 255.0
                        color_rgb = (r, g, b)
                    else:
                        color_rgb = tuple(raw_color)

                    # Fonte
                    font_raw = span.get("font", "helv")
                    is_bold = any(x in font_raw.lower() for x in
                                  ("bold", "bd", "heavy", "black", "semibold"))
                    is_italic = any(x in font_raw.lower() for x in
                                    ("italic", "oblique", "it", "slant"))
                    family = self._detect_family(font_raw)

                    # Alinhamento por posição na página
                    pw = page.rect.width
                    bbox = span["bbox"]
                    mid_x = (bbox[0] + bbox[2]) / 2
                    if abs(mid_x - pw / 2) < pw * 0.12:
                        align = "center"
                    elif bbox[0] > pw * 0.6:
                        align = "right"
                    else:
                        align = "left"

                    tb = TextBlock(
                        id=bid,
                        page=page_idx,
                        text=text,
                        x0=bbox[0], y0=bbox[1],
                        x1=bbox[2], y1=bbox[3],
                        font_size=span.get("size", 12.0),
                        font_name=font_raw,
                        font_family=family,
                        is_bold=is_bold,
                        is_italic=is_italic,
                        color_rgb=color_rgb,
                        align=align,
                        source="pymupdf",
                    )
                    result.append(tb)

        return result

    # ── OCR via Groq Vision ────────────────────────────────────────────────

    def extract_via_ocr(self, page_idx: int,
                         page_image_np=None) -> list[TextBlock]:
        """
        Usa Groq Vision para fazer OCR na página inteira (PDF escaneado).
        Retorna blocos com posições aproximadas.
        """
        if not self.groq_api_key:
            print("[Extractor] Groq não configurado — OCR não disponível.")
            return []

        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.groq_api_key,
                base_url="https://api.groq.com/openai/v1"
            )

            # Converte página para base64
            if page_image_np is not None:
                b64 = _np_to_b64(page_image_np)
            else:
                b64 = _fitz_page_to_b64(self.doc[page_idx])

            prompt = """Você é um motor de OCR especializado em documentos PDF.

Analise esta imagem de página de PDF e extraia TODOS os textos visíveis.

Para cada bloco de texto retorne um JSON com esta estrutura exata:
{
  "blocks": [
    {
      "text": "texto exato",
      "x_pct": 10.5,   // posição X como % da largura da página (0-100)
      "y_pct": 5.2,    // posição Y como % da altura da página (0-100)
      "w_pct": 30.0,   // largura como % da largura da página
      "h_pct": 3.5,    // altura como % da altura da página
      "font_size_pct": 2.1,  // tamanho da fonte como % da altura da página
      "is_bold": false,
      "is_italic": false,
      "font_family": "serif",  // serif | sans-serif | monospace
      "align": "left",         // left | center | right
      "color_r": 0, "color_g": 0, "color_b": 0
    }
  ]
}

Retorne APENAS o JSON válido, sem markdown, sem explicações."""

            resp = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ]
                }],
                temperature=0.05,
                max_tokens=4000,
            )

            raw = resp.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()

            import re
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                print("[Extractor OCR] JSON não encontrado na resposta")
                return []

            data = json.loads(m.group(0))
            page = self.doc[page_idx]
            pw, ph = page.rect.width, page.rect.height

            result = []
            for i, b in enumerate(data.get("blocks", [])):
                text = b.get("text", "").strip()
                if not text:
                    continue

                x0 = pw * b.get("x_pct", 0) / 100
                y0 = ph * b.get("y_pct", 0) / 100
                w  = pw * b.get("w_pct", 20) / 100
                h  = ph * b.get("h_pct", 3) / 100
                fs = ph * b.get("font_size_pct", 2) / 100

                r = b.get("color_r", 0) / 255.0
                g = b.get("color_g", 0) / 255.0
                bv = b.get("color_b", 0) / 255.0

                tb = TextBlock(
                    id=f"p{page_idx}_ocr_{i}",
                    page=page_idx,
                    text=text,
                    x0=x0, y0=y0,
                    x1=x0 + w, y1=y0 + h,
                    font_size=max(6.0, fs),
                    font_name=b.get("font_family", "helv"),
                    font_family=b.get("font_family", "sans-serif"),
                    is_bold=b.get("is_bold", False),
                    is_italic=b.get("is_italic", False),
                    color_rgb=(r, g, bv),
                    align=b.get("align", "left"),
                    source="groq_ocr",
                )
                result.append(tb)

            return result

        except Exception as e:
            print(f"[Extractor OCR] Erro: {e}")
            return []

    # ── Confirmação visual Groq Vision ─────────────────────────────────────

    def confirm_with_vision(self, blocks: list[TextBlock],
                             page_idx: int,
                             page_image_np=None) -> list[TextBlock]:
        """
        Manda amostra dos blocos + imagem da página para o Groq confirmar
        fonte, bold, italic e cor de cada span.
        Só processa blocos com fonte incerta (ex: nome genérico).
        """
        if not self.groq_api_key or not blocks:
            return blocks

        # Filtra blocos que precisam de confirmação (fonte genérica)
        uncertain = [b for b in blocks
                     if b.font_name.lower() in ("helv", "arial", "times",
                                                "cour", "helvetica")]
        if not uncertain:
            return blocks

        # Limita a 20 blocos por chamada
        sample = uncertain[:20]

        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.groq_api_key,
                base_url="https://api.groq.com/openai/v1"
            )

            if page_image_np is not None:
                b64 = _np_to_b64(page_image_np)
            else:
                b64 = _fitz_page_to_b64(self.doc[page_idx])

            # Monta lista de textos para a IA confirmar
            texts_desc = "\n".join(
                f'{i}. "{b.text[:40]}" (font: {b.font_name}, size: {b.font_size:.1f}pt)'
                for i, b in enumerate(sample)
            )

            prompt = f"""Analise esta página de PDF e confirme as propriedades tipográficas dos textos abaixo.

Textos para confirmar:
{texts_desc}

Para cada texto, retorne um JSON array com as confirmações:
[
  {{
    "index": 0,
    "font_name": "times ou arial ou calibri ou verdana ou courier",
    "font_family": "serif | sans-serif | monospace",
    "is_bold": true/false,
    "is_italic": true/false,
    "color_r": 0, "color_g": 0, "color_b": 0
  }}
]

Retorne APENAS o JSON array válido."""

            resp = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ]
                }],
                temperature=0.05,
                max_tokens=2000,
            )

            raw = resp.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()

            import re
            m = re.search(r'\[.*\]', raw, re.DOTALL)
            if not m:
                return blocks

            corrections = json.loads(m.group(0))
            corr_map = {c["index"]: c for c in corrections}

            # Aplica correções nos blocos
            for i, b in enumerate(sample):
                if i in corr_map:
                    c = corr_map[i]
                    b.font_name    = c.get("font_name", b.font_name)
                    b.font_family  = c.get("font_family", b.font_family)
                    b.is_bold      = c.get("is_bold", b.is_bold)
                    b.is_italic    = c.get("is_italic", b.is_italic)
                    r = c.get("color_r", int(b.color_rgb[0]*255)) / 255.0
                    g = c.get("color_g", int(b.color_rgb[1]*255)) / 255.0
                    bv = c.get("color_b", int(b.color_rgb[2]*255)) / 255.0
                    b.color_rgb    = (r, g, bv)
                    b.source       = "groq_vision"

            return blocks

        except Exception as e:
            print(f"[Extractor Vision] Erro: {e}")
            return blocks

    # ── Extração completa de uma página ───────────────────────────────────

    def extract_page(self, page_idx: int,
                      use_vision_confirm: bool = False,
                      page_image_np=None,
                      progress_cb=None) -> list[TextBlock]:
        """
        Extração completa de uma página.
        - Detecta modo (native / scanned)
        - Extrai via PyMuPDF ou OCR
        - Opcionalmente confirma com Groq Vision
        """
        if page_idx in self._blocks_cache:
            return self._blocks_cache[page_idx]

        if progress_cb:
            progress_cb(f"Analisando página {page_idx + 1}...")

        mode = self.detect_page_mode(page_idx)

        if mode == "scanned":
            if not self.groq_api_key:
                # Sem Groq, tenta extrair nativo mesmo assim (pode retornar vazio)
                if progress_cb:
                    progress_cb(f"Página {page_idx+1} escaneada — configure Groq para OCR.")
                blocks = self.extract_native(page_idx)
            else:
                if progress_cb:
                    progress_cb(f"Página {page_idx+1} escaneada — usando OCR...")
                blocks = self.extract_via_ocr(page_idx, page_image_np)
        else:
            if progress_cb:
                progress_cb(f"Extraindo texto nativo da página {page_idx+1}...")
            blocks = self.extract_native(page_idx)

        if use_vision_confirm and blocks and self.groq_api_key:
            if progress_cb:
                progress_cb(f"Confirmando fontes com Groq Vision...")
            blocks = self.confirm_with_vision(blocks, page_idx, page_image_np)

        self._blocks_cache[page_idx] = blocks
        return blocks

    def extract_all_pages(self, use_vision_confirm: bool = False,
                           progress_cb=None) -> dict[int, list[TextBlock]]:
        """Extrai todas as páginas."""
        result = {}
        for i in range(self.page_count):
            result[i] = self.extract_page(i, use_vision_confirm,
                                           progress_cb=progress_cb)
        return result

    def invalidate_cache(self, page_idx: int):
        """Invalida cache de uma página (após edição)."""
        self._blocks_cache.pop(page_idx, None)

    def update_block(self, block_id: str, new_text: str) -> Optional[TextBlock]:
        """Atualiza o texto editado de um bloco pelo ID."""
        for page_blocks in self._blocks_cache.values():
            for b in page_blocks:
                if b.id == block_id:
                    b.edited_text = new_text
                    return b
        return None

    def get_block(self, block_id: str) -> Optional[TextBlock]:
        for page_blocks in self._blocks_cache.values():
            for b in page_blocks:
                if b.id == block_id:
                    return b
        return None

    def get_edited_blocks(self) -> list[TextBlock]:
        """Retorna todos os blocos que foram editados."""
        result = []
        for page_blocks in self._blocks_cache.values():
            for b in page_blocks:
                if b.is_edited:
                    result.append(b)
        return result

    # ── Helpers ────────────────────────────────────────────────────────────

    def _detect_family(self, font_name: str) -> str:
        fn = font_name.lower()
        if any(x in fn for x in ("cour", "mono", "fixed", "consol", "code")):
            return "monospace"
        if any(x in fn for x in ("tim", "roman", "serif", "georgia",
                                  "garamond", "palatino", "tiro")):
            return "serif"
        return "sans-serif"


# ── Helpers de imagem ──────────────────────────────────────────────────────

def _np_to_b64(img_np, max_dim: int = 1200) -> str:
    """Converte numpy array para base64 PNG."""
    try:
        import numpy as np
        from PIL import Image
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            try:
                import cv2
                rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            except Exception:
                rgb = img_np[:, :, ::-1]
        else:
            rgb = img_np

        pil = Image.fromarray(rgb.astype("uint8"))
        w, h = pil.size
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            pil = pil.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[_np_to_b64] {e}")
        return ""


def _fitz_page_to_b64(page: fitz.Page, max_dim: int = 1200) -> str:
    """Renderiza página fitz para base64."""
    from PIL import Image
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
