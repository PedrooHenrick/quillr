"""
backend/start.py — carrega .env e inicia uvicorn
"""
import os
import sys
from pathlib import Path

# Carrega .env se existir (desenvolvimento local)
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

# Adiciona Poppler ao PATH se existir (Windows)
poppler = Path(__file__).parent / 'poppler' / 'bin'
if poppler.exists():
    os.environ['PATH'] = str(poppler) + os.pathsep + os.environ.get('PATH', '')

# ── Porta: Railway injeta a variável PORT automaticamente ──────────────────
# Em desenvolvimento local, usa 8000 como fallback.
port = int(os.environ.get("PORT", 8000))

# Inicia
import uvicorn
uvicorn.run(
    'main:app',
    host='0.0.0.0',   # obrigatório para o Railway conseguir acessar
    port=port,
    reload=False,      # reload=True não pode ser usado em produção
)
