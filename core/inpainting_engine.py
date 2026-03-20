"""
core/inpainting_engine.py — completo
Motor de edição de PDF usando inpainting do OpenCV.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

DEFAULT_DPI    = 200
INPAINT_RADIUS = 5


# ── Conversão ──────────────────────────────────────────────────────────────

def pdf_page_to_image(pdf_path: str, page_number: int = 0,
                      dpi: int = DEFAULT_DPI) -> "np.ndarray":
    """Converte uma página do PDF em imagem NumPy BGR."""
    images = convert_from_path(
        pdf_path, dpi=dpi,
        first_page=page_number + 1,
        last_page=page_number + 1,
    )
    if not images:
        raise ValueError(f"Não foi possível converter a página {page_number}.")
    rgb = np.array(images[0])
    if CV2_AVAILABLE:
        import cv2
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb


def pdf_all_pages_to_images(pdf_path: str,
                              dpi: int = DEFAULT_DPI) -> list:
    """Converte todas as páginas do PDF em lista de imagens NumPy BGR."""
    images = convert_from_path(pdf_path, dpi=dpi)
    result = []
    for img in images:
        arr = np.array(img)
        if CV2_AVAILABLE:
            import cv2
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        result.append(arr)
    return result


def image_to_pdf(images: list, output_path: str,
                 dpi: int = DEFAULT_DPI) -> str:
    """Converte lista de imagens NumPy em PDF multipágina."""
    pil_images = []
    for img in images:
        if CV2_AVAILABLE:
            import cv2
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb = img
        pil_images.append(Image.fromarray(rgb))

    if not pil_images:
        raise ValueError("Nenhuma imagem fornecida.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    first = pil_images[0]
    rest  = pil_images[1:]
    first.save(output_path, format="PDF", save_all=True,
               append_images=rest, resolution=dpi)
    return output_path


# ── Substituição nativa de texto (preserva PDF vetorial) ──────────────────

def save_text_edits_native(pdf_path: str, edits: list, output_path: str):
    """
    Substitui texto diretamente no PDF usando PyMuPDF.
    Preserva o PDF vetorial — NÃO converte para imagem.
    O PDF continua editável após salvar.

    edits: lista de dicts com:
      - page (int)
      - x0, y0, x1, y1 (float — coordenadas em pontos PDF)
      - new_text (str)
      - font_name (str)
      - font_size (float)
      - color_rgb (list [r, g, b] de 0.0 a 1.0)
      - align (str: "left" | "center" | "right")
    """
    import fitz

    doc = fitz.open(pdf_path)

    by_page = {}
    for e in edits:
        by_page.setdefault(e["page"], []).append(e)

    for page_idx, page_edits in by_page.items():
        page = doc[page_idx]

        for edit in page_edits:
            x0 = edit["x0"]
            y0 = edit["y0"]
            x1 = edit["x1"]
            y1 = edit["y1"]
            new_text  = edit["new_text"]
            r, g, b   = edit.get("color_rgb", [0.0, 0.0, 0.0])
            font_size = float(edit.get("font_size", 12.0))
            align_str = edit.get("align", "left")
            align_map = {"left": 0, "center": 1, "right": 2}
            align     = align_map.get(align_str, 0)

            rect = fitz.Rect(x0, y0, x1, y1)

            # 1. Apaga o texto original com retângulo branco
            page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))

            # 2. Insere o novo texto vetorial na mesma área
            page.insert_textbox(
                rect,
                new_text,
                fontsize=font_size,
                color=(r, g, b),
                align=align,
            )

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()


# ── Inpainting ─────────────────────────────────────────────────────────────

def build_mask_from_area(image, x1: int, y1: int, x2: int, y2: int,
                          threshold: int = 80) -> "np.ndarray":
    """Cria máscara binária detectando texto escuro na área."""
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV não instalado. Rode: pip install opencv-python")
    import cv2
    roi  = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask_roi = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_roi = cv2.dilate(mask_roi, kernel, iterations=1)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = mask_roi
    return mask


def build_mask_full_area(x1: int, y1: int, x2: int, y2: int,
                          img_shape: tuple) -> "np.ndarray":
    """Máscara sólida cobrindo toda a área."""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def inpaint_area(image, mask, radius: int = INPAINT_RADIUS):
    """Aplica cv2.inpaint() reconstruindo fundo sem cor artificial."""
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV não instalado. Rode: pip install opencv-python")
    import cv2
    return cv2.inpaint(image, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


def remove_content_inpaint(image, x1: int, y1: int, x2: int, y2: int,
                             threshold: int = 80, full_area: bool = False,
                             radius: int = INPAINT_RADIUS):
    """
    Remove conteúdo da área usando inpainting — reconstrói o fundo.
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV não instalado.")
    import cv2

    mask = build_mask_full_area(x1, y1, x2, y2, image.shape)
    result = inpaint_area(image, mask, radius=radius)

    y1e = max(0, y1 - 2)
    y2e = min(image.shape[0], y2 + 2)
    x1e = max(0, x1 - 2)
    x2e = min(image.shape[1], x2 + 2)
    mask2 = build_mask_full_area(x1e, y1e, x2e, y2e, image.shape)
    result = inpaint_area(result, mask2, radius=max(3, radius - 2))

    return result


# ── Inserção ───────────────────────────────────────────────────────────────

FONT_MAP = {
    "times":    ["C:/Windows/Fonts/times.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"],
    "timesbd":  ["C:/Windows/Fonts/timesbd.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"],
    "timesi":   ["C:/Windows/Fonts/timesi.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"],
    "arial":    ["C:/Windows/Fonts/arial.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"],
    "arialbd":  ["C:/Windows/Fonts/arialbd.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"],
    "calibri":  ["C:/Windows/Fonts/calibri.ttf"],
    "calibrib": ["C:/Windows/Fonts/calibrib.ttf"],
    "verdana":  ["C:/Windows/Fonts/verdana.ttf"],
    "tahoma":   ["C:/Windows/Fonts/tahoma.ttf"],
    "courier":  ["C:/Windows/Fonts/cour.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"],
    "default":  ["C:/Windows/Fonts/arial.ttf",
                 "C:/Windows/Fonts/times.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"],
}


def _resolve_font(fontname_hint: str, size: int) -> "ImageFont.FreeTypeFont":
    hint = fontname_hint.lower()
    is_bold   = any(x in hint for x in ("bold","bd","black","heavy","semibold"))
    is_italic = any(x in hint for x in ("italic","it","oblique","slant"))
    candidates = []

    if any(x in hint for x in ("times","tiro","roman","serif")):
        key = "timesbd" if is_bold else ("timesi" if is_italic else "times")
        candidates += FONT_MAP.get(key, []) + FONT_MAP["times"]
    elif any(x in hint for x in ("calibri",)):
        key = "calibrib" if is_bold else "calibri"
        candidates += FONT_MAP.get(key, []) + FONT_MAP["calibri"]
    elif any(x in hint for x in ("arial","helv","helvetica","sans")):
        key = "arialbd" if is_bold else "arial"
        candidates += FONT_MAP.get(key, []) + FONT_MAP["arial"]
    elif any(x in hint for x in ("cour","mono","courier")):
        candidates += FONT_MAP["courier"]
    elif any(x in hint for x in ("verdana",)):
        candidates += FONT_MAP["verdana"]
    elif any(x in hint for x in ("tahoma",)):
        candidates += FONT_MAP["tahoma"]

    candidates += FONT_MAP["default"]

    for path in candidates:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue

    return ImageFont.load_default()


def _fit_font_to_area(text: str, font_path_or_hint: str,
                       area_w: int, area_h: int,
                       start_size: int) -> tuple:
    size = min(start_size, area_h - 2)
    size = max(6, size)
    font = _resolve_font(font_path_or_hint, size)
    return font, size


def smart_replace_text(image, new_text: str,
                        x1: int, y1: int, x2: int, y2: int,
                        original_text: str = "",
                        fontname_hint: str = "arial",
                        font_size_hint: int = 24,
                        color_bgr: tuple = (0, 0, 0),
                        align: str = "left",
                        font_path_override: str | None = None):
    area_w = x2 - x1
    area_h = y2 - y1

    if area_w <= 0 or area_h <= 0:
        return image

    adjusted_start = font_size_hint
    hint = font_path_override if font_path_override else fontname_hint
    font, final_size = _fit_font_to_area(
        new_text, hint, area_w, area_h, adjusted_start)

    try:
        bbox = font.getbbox(new_text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        try:
            tw, th = font.getsize(new_text)
        except Exception:
            tw = len(new_text) * final_size // 2
            th = final_size

    if align == "center":
        tx = x1 + (area_w - tw) // 2
    elif align == "right":
        tx = x2 - tw
    else:
        tx = x1

    ty = y1 + max(0, (area_h - th) // 2)

    if CV2_AVAILABLE:
        import cv2
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = image.copy()

    pil  = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

    try:
        draw.text((tx, ty), new_text, fill=color_rgb, font=font, anchor="lt")
    except TypeError:
        draw.text((tx, ty), new_text, fill=color_rgb, font=font)

    result = np.array(pil)
    if CV2_AVAILABLE:
        import cv2
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    return result


def insert_text_on_image(image, text: str, x: int, y: int,
                          font_size: int = 24,
                          color_bgr: tuple = (0, 0, 0),
                          font_path: str | None = None):
    if CV2_AVAILABLE:
        import cv2
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = image.copy()

    pil  = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = _resolve_font(font_path or "arial", font_size)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    try:
        draw.text((x, y), text, fill=color_rgb, font=font, anchor="lt")
    except TypeError:
        draw.text((x, y), text, fill=color_rgb, font=font)

    result = np.array(pil)
    if CV2_AVAILABLE:
        import cv2
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    return result


def insert_image_on_image(base_image, overlay_path: str,
                           x1: int, y1: int, x2: int, y2: int):
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV não instalado.")
    import cv2
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        raise FileNotFoundError(f"Imagem não encontrada: {overlay_path}")
    w, h   = x2 - x1, y2 - y1
    result = base_image.copy()
    ov     = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)
    if ov.shape[2] == 4:
        alpha = ov[:, :, 3:4] / 255.0
        roi   = result[y1:y2, x1:x2].astype(float)
        blend = roi * (1 - alpha) + ov[:, :, :3].astype(float) * alpha
        result[y1:y2, x1:x2] = blend.astype(np.uint8)
    else:
        result[y1:y2, x1:x2] = ov[:, :, :3]
    return result


# ── Classe principal ───────────────────────────────────────────────────────

class InpaintingEngine:
    def __init__(self, pdf_path: str, dpi: int = DEFAULT_DPI):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")
        self.pdf_path  = pdf_path
        self.dpi       = dpi
        print(f"[InpaintingEngine] Carregando '{pdf_path}' @ {dpi} DPI...")
        self.pages     = pdf_all_pages_to_images(pdf_path, dpi=dpi)
        self._original = [p.copy() for p in self.pages]
        print(f"[InpaintingEngine] {len(self.pages)} página(s) carregada(s).")

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def get_page_image(self, page: int):
        self._check(page)
        return self.pages[page]

    def reset_page(self, page: int):
        self._check(page)
        self.pages[page] = self._original[page].copy()

    def remove_content(self, page: int, x1: int, y1: int, x2: int, y2: int,
                        threshold: int = 80, full_area: bool = False,
                        inpaint_radius: int = INPAINT_RADIUS) -> "InpaintingEngine":
        self._check(page)
        self.pages[page] = remove_content_inpaint(
            self.pages[page], x1, y1, x2, y2,
            threshold=threshold, full_area=full_area, radius=inpaint_radius)
        return self

    def add_text(self, page: int, text: str, x: int, y: int,
                 font_size: int = 24, color_bgr: tuple = (0, 0, 0),
                 font_path: str | None = None) -> "InpaintingEngine":
        self._check(page)
        self.pages[page] = insert_text_on_image(
            self.pages[page], text, x, y,
            font_size=font_size, color_bgr=color_bgr, font_path=font_path)
        return self

    def add_signature(self, page: int, image_path: str,
                      x1: int, y1: int, x2: int, y2: int) -> "InpaintingEngine":
        self._check(page)
        self.pages[page] = insert_image_on_image(
            self.pages[page], image_path, x1, y1, x2, y2)
        return self

    def save(self, output_path: str) -> str:
        print(f"[save] → '{output_path}'")
        image_to_pdf(self.pages, output_path, dpi=self.dpi)
        return output_path

    @staticmethod
    def process_batch(pdf_list: list, operations: list,
                      output_dir: str = "output",
                      dpi: int = DEFAULT_DPI,
                      suffix: str = "_editado") -> list:
        os.makedirs(output_dir, exist_ok=True)
        outputs = []
        for pdf_path in pdf_list:
            try:
                engine = InpaintingEngine(pdf_path, dpi=dpi)
                for op in operations:
                    t    = op.get("type", "")
                    page = op.get("page", 0)
                    if t == "remove":
                        engine.remove_content(
                            page, op["x1"], op["y1"], op["x2"], op["y2"],
                            threshold=op.get("threshold", 80),
                            full_area=op.get("full_area", False),
                            inpaint_radius=op.get("radius", INPAINT_RADIUS))
                    elif t == "text":
                        engine.add_text(
                            page, op["text"], op["x"], op["y"],
                            font_size=op.get("font_size", 24),
                            color_bgr=op.get("color_bgr", (0,0,0)),
                            font_path=op.get("font_path"))
                    elif t == "signature":
                        engine.add_signature(
                            page, op["image_path"],
                            op["x1"], op["y1"], op["x2"], op["y2"])
                base     = os.path.splitext(os.path.basename(pdf_path))[0]
                out_path = os.path.join(output_dir, f"{base}{suffix}.pdf")
                engine.save(out_path)
                outputs.append(out_path)
            except Exception as e:
                print(f"[ERRO] '{pdf_path}': {e}")
        return outputs

    def _check(self, page: int):
        if not (0 <= page < self.page_count):
            raise IndexError(f"Página {page} inválida (total: {self.page_count})")
