"""
core/groq_ai.py
═══════════════════════════════════════════════════════════════════
Integração com Groq AI — análise VISUAL da área selecionada.

O sistema manda a imagem da área selecionada + página completa
para o Groq Vision analisar e retornar:
  - tamanho exato da fonte em pixels
  - família da fonte (serif/sans/mono)
  - negrito / itálico
  - cor do texto
  - alinhamento
  - proporção de escala para o texto novo caber igual ao original
═══════════════════════════════════════════════════════════════════
"""

import os
import json
import base64
import io

# ══════════════════════════════════════════════════════════════════
# ⚙️  CONFIGURE SUA API KEY AQUI
# ══════════════════════════════════════════════════════════════════
GROQ_API_KEY = ""   # ← Cole sua chave aqui  ex: "gsk_xxxxxxxxxxxx"
# ══════════════════════════════════════════════════════════════════

GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # suporta visão
GROQ_TEXT_MODEL   = "llama-3.3-70b-versatile"
GROQ_BASE_URL     = "https://api.groq.com/openai/v1"


def _get_api_key() -> str:
    if GROQ_API_KEY.strip():
        return GROQ_API_KEY.strip()
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GROQ_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("GROQ_API_KEY", "")


def is_configured() -> bool:
    return bool(_get_api_key())


def _numpy_to_base64(img_array) -> str:
    """Converte imagem NumPy para base64 PNG."""
    try:
        from PIL import Image
        import numpy as np
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # BGR → RGB
            try:
                import cv2
                rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            except Exception:
                rgb = img_array[:, :, ::-1]
        else:
            rgb = img_array
        pil = Image.fromarray(rgb.astype("uint8"))
        # Redimensiona se muito grande (máx 800px) para economizar tokens
        max_dim = 800
        w, h = pil.size
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            pil = pil.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[Groq] _numpy_to_base64 error: {e}")
        return ""


def analyze_image_area(
    area_image,           # np.ndarray — recorte da área selecionada
    page_image,           # np.ndarray — página completa
    original_text: str,
    new_text: str,
    area_w: int,
    area_h: int,
) -> dict:
    """
    Manda as imagens para o Groq Vision analisar visualmente.

    O modelo vê:
      1. A página completa — para entender o contexto do documento
      2. O recorte da área — para analisar a fonte com precisão

    Retorna dict com todas as propriedades necessárias para replicar
    o texto original perfeitamente.
    """
    api_key = _get_api_key()
    if not api_key:
        return _fallback(original_text, new_text, area_h)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

        # Converte imagens para base64
        b64_area = _numpy_to_base64(area_image)
        b64_page = _numpy_to_base64(page_image)

        if not b64_area or not b64_page:
            return _fallback(original_text, new_text, area_h)

        prompt = f"""Você é um especialista em análise tipográfica de documentos PDF.

Analise as duas imagens fornecidas:
- Imagem 1: página completa do documento
- Imagem 2: recorte ampliado da área onde está o texto "{original_text}"

Quero substituir "{original_text}" por "{new_text}".
A área disponível tem {area_w}px de largura e {area_h}px de altura.

Analise visualmente e retorne APENAS um JSON válido (sem markdown):
{{
    "font_family": "serif ou sans-serif ou monospace",
    "font_name": "times ou arial ou calibri ou courier ou verdana",
    "is_bold": true ou false,
    "is_italic": true ou false,
    "font_size_px": <tamanho REAL da fonte em pixels que você vê na imagem — este é o tamanho que deve ser preservado no texto novo>,
    "color_r": <0-255>,
    "color_g": <0-255>,
    "color_b": <0-255>,
    "align": "left ou center ou right",
    "scale_factor": 1.0,
    "notes": "observação breve"
}}

Regras IMPORTANTES:
- font_size_px deve ser o tamanho REAL dos caracteres na imagem ({area_h}px é a altura total da área)
- Estime font_size_px como aproximadamente 75-85% da altura da área, ou seja, cerca de {int(area_h * 0.80)}px
- NÃO ajuste o tamanho pelo comprimento do texto novo — preserve o tamanho original
- scale_factor deve ser SEMPRE 1.0 — não altere o tamanho pela quantidade de caracteres
- Se o texto parece Times New Roman / serifado → font_name = "times"
- Se parece Arial/Helvetica → font_name = "arial"
- Analise a cor real do texto na imagem"""

        # Tenta modelo de visão primeiro, fallback para texto
        try:
            response = client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_page}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_area}"},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }],
                temperature=0.1,
                max_tokens=400,
            )
        except Exception as vision_err:
            print(f"[Groq] Modelo de visão falhou ({vision_err}), usando modelo de texto...")
            # Fallback: modelo texto com descrição das propriedades detectadas
            text_prompt = f"""Analise propriedades tipográficas para substituição de texto em PDF.

Texto original: "{original_text}" ({len(original_text)} caracteres)
Texto novo: "{new_text}" ({len(new_text)} caracteres)
Área disponível: {area_w}px largura x {area_h}px altura

Retorne APENAS JSON válido:
{{
    "font_family": "serif",
    "font_name": "times",
    "is_bold": false,
    "is_italic": false,
    "font_size_px": {max(8, int(area_h * 0.80))},
    "color_r": 0,
    "color_g": 0,
    "color_b": 0,
    "align": "left",
    "scale_factor": 1.0,
    "notes": "análise por texto"
}}

IMPORTANTE: font_size_px deve preservar o tamanho original — NÃO reduza pelo comprimento do novo texto."""
            response = client.chat.completions.create(
                model=GROQ_TEXT_MODEL,
                messages=[{"role": "user", "content": text_prompt}],
                temperature=0.1,
                max_tokens=300,
            )

        raw = response.choices[0].message.content.strip()
        print(f"[Groq raw response]: {repr(raw[:500])}")

        # Limpa markdown e extrai JSON
        raw = raw.replace("```json", "").replace("```", "").strip()

        # Tenta encontrar o bloco JSON dentro da resposta
        import re
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        
        if not raw:
            print("[Groq] Resposta vazia do modelo, usando fallback")
            return _fallback(original_text, new_text, area_h)

        result = json.loads(raw)

        # ── Calcula tamanho final da fonte ──────────────────────────
        # REGRA CORRETA:
        #   Preservar o tamanho ORIGINAL da fonte detectada visualmente.
        #   NÃO escalar pelo número de caracteres — isso causava texto pequeno
        #   quando o novo texto era mais longo que o original.
        #   O font_size_px do Groq é o tamanho real detectado na imagem.

        # Tamanho detectado pelo Groq Vision (tamanho real dos caracteres)
        groq_size = int(result.get("font_size_px", 0))

        # Se o Groq detectou um tamanho válido, usa ele diretamente
        if groq_size >= 8:
            final_size = groq_size
        else:
            # Fallback: 78% da altura da área
            final_size = max(8, int(area_h * 0.78))

        # Nunca ultrapassa a altura da área
        final_size = min(final_size, area_h - 2)

        # scale_factor ainda é retornado mas NÃO afeta o tamanho da fonte
        scale = float(result.get("scale_factor", 1.0))

        print(f"[Groq] area_h={area_h}px | groq_detected={groq_size}px | final_size={final_size}px")

        r = int(result.get("color_r", 0))
        g = int(result.get("color_g", 0))
        b = int(result.get("color_b", 0))

        notes = result.get("notes", "")
        print(f"[Groq Vision] {notes} | fonte: {result.get('font_name')} "
              f"{'bold' if result.get('is_bold') else ''} "
              f"{'italic' if result.get('is_italic') else ''} "
              f"| size: {final_size}px | align: {result.get('align')} "
              f"| cor: rgb({r},{g},{b})")

        return {
            "fontname":   result.get("font_name", "arial"),
            "is_bold":    bool(result.get("is_bold", False)),
            "is_italic":  bool(result.get("is_italic", False)),
            "font_size_px": final_size,
            "color_bgr":  (b, g, r),   # PIL usa RGB, OpenCV BGR
            "color_rgb":  (r/255, g/255, b/255),
            "align":      result.get("align", "left"),
            "scale_factor": scale,
            "notes":      notes,
            "success":    True,
        }

    except ImportError:
        print("[Groq] openai não instalado. Rode: pip install openai")
        return _fallback(original_text, new_text, area_h)
    except json.JSONDecodeError as e:
        print(f"[Groq] JSON inválido: {e}")
        return _fallback(original_text, new_text, area_h)
    except Exception as e:
        print(f"[Groq] Erro: {e}")
        return _fallback(original_text, new_text, area_h)


def _fallback(original_text: str, new_text: str, area_h: int) -> dict:
    """Análise local quando Groq não disponível."""
    # NÃO escala pelo número de caracteres — preserva tamanho baseado na área
    fs = max(8, int(area_h * 0.80))
    fs = min(fs, area_h)
    return {
        "fontname":     "arial",
        "is_bold":      False,
        "is_italic":    False,
        "font_size_px": fs,
        "color_bgr":    (0, 0, 0),
        "color_rgb":    (0.0, 0.0, 0.0),
        "align":        "left",
        "scale_factor": 1.0,
        "notes":        "Análise local (Groq não configurado)",
        "success":      False,
    }


def get_font_path_for(fontname: str, bold: bool = False,
                       italic: bool = False) -> str | None:
    """Retorna o caminho TTF para a fonte especificada."""
    base = fontname.lower()
    table = {
        ("times",   False, False): ["C:/Windows/Fonts/times.ttf",
                                    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"],
        ("times",   True,  False): ["C:/Windows/Fonts/timesbd.ttf",
                                    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"],
        ("times",   False, True):  ["C:/Windows/Fonts/timesi.ttf",
                                    "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"],
        ("times",   True,  True):  ["C:/Windows/Fonts/timesbi.ttf"],
        ("arial",   False, False): ["C:/Windows/Fonts/arial.ttf",
                                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"],
        ("arial",   True,  False): ["C:/Windows/Fonts/arialbd.ttf",
                                    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"],
        ("arial",   False, True):  ["C:/Windows/Fonts/ariali.ttf"],
        ("arial",   True,  True):  ["C:/Windows/Fonts/arialbi.ttf"],
        ("calibri", False, False): ["C:/Windows/Fonts/calibri.ttf"],
        ("calibri", True,  False): ["C:/Windows/Fonts/calibrib.ttf"],
        ("calibri", False, True):  ["C:/Windows/Fonts/calibrii.ttf"],
        ("calibri", True,  True):  ["C:/Windows/Fonts/calibriz.ttf"],
        ("courier", False, False): ["C:/Windows/Fonts/cour.ttf"],
        ("courier", True,  False): ["C:/Windows/Fonts/courbd.ttf"],
        ("verdana", False, False): ["C:/Windows/Fonts/verdana.ttf"],
        ("verdana", True,  False): ["C:/Windows/Fonts/verdanab.ttf"],
        ("tahoma",  False, False): ["C:/Windows/Fonts/tahoma.ttf"],
    }
    paths = table.get((base, bold, italic), [])
    if not paths:
        paths = table.get((base, False, False), [])
    if not paths:
        paths = table.get(("arial", False, False), [])
    for p in paths:
        if os.path.exists(p):
            return p
    return None
