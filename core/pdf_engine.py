"""
core/pdf_engine.py — versao web (sem PyQt6)
"""
import fitz
import os


class PDFEngine:

    def __init__(self):
        self.doc = None
        self.file_path = ""
        self._modified = False

    def open(self, path: str) -> bool:
        try:
            if self.doc:
                self.doc.close()
            self.doc = fitz.open(path)
            self.file_path = path
            self._modified = False
            return True
        except Exception as e:
            print(f"open error: {e}"); return False

    def close(self):
        if self.doc:
            self.doc.close(); self.doc = None

    @property
    def page_count(self): return self.doc.page_count if self.doc else 0
    @property
    def is_open(self): return self.doc is not None
    @property
    def is_modified(self): return self._modified

    def get_page_size(self, page_index: int):
        if not self.doc or page_index >= self.page_count: return (0.0, 0.0)
        r = self.doc[page_index].rect
        return (r.width, r.height)

    def get_text_at_rect(self, page_index: int, rect: tuple):
        if not self.doc: return None
        try:
            page = self.doc[page_index]
            clip = fitz.Rect(*rect)
            blocks = page.get_text("dict", clip=clip)["blocks"]
            best_span = None; best_area = 0.0
            for block in blocks:
                if block.get("type") != 0: continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        sr = fitz.Rect(span["bbox"])
                        area = abs(sr & clip)
                        if area > best_area:
                            best_area = area; best_span = span
            if best_span:
                raw = best_span.get("color", 0)
                if isinstance(raw, int):
                    color_rgb = (((raw>>16)&0xFF)/255.0, ((raw>>8)&0xFF)/255.0, (raw&0xFF)/255.0)
                else:
                    color_rgb = tuple(raw)
                return {
                    "text": best_span.get("text","").strip(),
                    "size": best_span.get("size", 11.0),
                    "fontname": best_span.get("font","helv"),
                    "color": raw, "color_rgb": color_rgb,
                    "bbox": best_span["bbox"],
                }
        except Exception as e:
            print(f"get_text_at_rect error: {e}")
        return None

    def save(self, path=None) -> bool:
        if not self.doc: return False
        target = path or self.file_path
        try:
            if path and path != self.file_path:
                self.doc.save(target, garbage=4, deflate=True)
            else:
                tmp = target + ".tmp"
                self.doc.save(tmp, garbage=4, deflate=True)
                os.replace(tmp, target)
            self.file_path = target; self._modified = False
            return True
        except Exception as e:
            print(f"save error: {e}"); return False
