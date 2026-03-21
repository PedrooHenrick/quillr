"""
core/extractor.py
═══════════════════════════════════════════════════════════════════
Motor de extração de texto do PDF.

Estratégia:
  1. PyMuPDF extrai todos os blocos de texto nativos (posição, fonte, tamanho, cor)
  2. Se não achar texto nativo (página virou imagem), lê a camada invisível
     que o /save-text injeta após cada edição — sem OCR externo.
  3. Retorna lista de TextBlock com tudo que o editor precisa
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
    id: str
    page: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float = 12.0
    font_name: str = "helv"
    font_family: str = "sans-serif"
    is_bold: bool = False
    is_italic: bool = False
    color_rgb: tuple = (0.0, 0.0, 0.0)
    align: str = "left"
    source: str = "pymupdf"
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
    Usa PyMuPDF nativo — sem OCR externo.
    Após edição de texto, lê a camada invisível injetada pelo /save-text.
    """

    def __init__(self, groq_api_key: str = ""):
        # groq_api_key mantido por compatibilidade mas não usado para OCR
        self.groq_api_key = groq_api_key
        self._blocks_cache: dict[int, list[TextBlock]] = {}
        self.doc: Optional[fitz.Document] = None
        self.file_path: str = ""
        self.page_count: int = 0
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
        """
        Detecta se a página tem texto nativo ou virou imagem.
        Se virou imagem mas tem camada invisível injetada, retorna 'overlay'.
        """
        if page_idx in self.page_modes:
            return self.page_modes[page_idx]
        if not self.doc:
            return "unknown"

        page = self.doc[page_idx]
        blocks = page.get_text("dict")["blocks"]
        span_count = sum(
            len(line.get("spans", []))
            for b in blocks if b.get("type") == 0
            for line in b.get("lines", [])
        )

        if span_count > 0:
            mode = "native"
        else:
            mode = "image_only"

        self.page_modes[page_idx] = mode
        return mode

    # ── Extração PyMuPDF nativa ────────────────────────────────────────────

    def extract_native(self, page_idx: int) -> list[TextBlock]:
        """Extrai texto nativo via PyMuPDF — rápido e preciso em posição."""
        if not self.doc:
            return []
        page = self.doc[page_idx]
        blocks_raw = page.get_text("dict")["blocks"]
        result = []

        for bi, block in enumerate(blocks_raw):
            if block.get("type") != 0:
                continue
            for li, line in enumerate(block.get("lines", [])):
                for si, span in enumerate(line.get("spans", [])):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

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

    # ── Extração completa de uma página ───────────────────────────────────

    def extract_page(self, page_idx: int,
                      use_vision_confirm: bool = False,
                      page_image_np=None,
                      progress_cb=None) -> list[TextBlock]:
        """
        Extração completa de uma página.
        - Se tem texto nativo → extrai direto
        - Se virou imagem → lê camada invisível injetada pelo /save-text
        - Sem OCR externo
        """
        if page_idx in self._blocks_cache:
            return self._blocks_cache[page_idx]

        if progress_cb:
            progress_cb(f"Analisando página {page_idx + 1}...")

        mode = self.detect_page_mode(page_idx)

        if mode == "native":
            if progress_cb:
                progress_cb(f"Extraindo texto nativo da página {page_idx + 1}...")
            blocks = self.extract_native(page_idx)
        else:
            # Página virou imagem — tenta ler camada invisível
            if progress_cb:
                progress_cb(f"Lendo camada de texto da página {page_idx + 1}...")
            blocks = self.extract_native(page_idx)
            if not blocks:
                print(f"[Extractor] p{page_idx}: imagem sem camada de texto — sem blocos.")

        self._blocks_cache[page_idx] = blocks
        return blocks

    def extract_all_pages(self, use_vision_confirm: bool = False,
                           progress_cb=None) -> dict[int, list[TextBlock]]:
        result = {}
        for i in range(self.page_count):
            result[i] = self.extract_page(i, progress_cb=progress_cb)
        return result

    def invalidate_cache(self, page_idx: int):
        self._blocks_cache.pop(page_idx, None)

    def update_block(self, block_id: str, new_text: str) -> Optional[TextBlock]:
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
