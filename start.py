"""
backend/start.py — carrega .env e inicia uvicorn
"""
import os
import sys
from pathlib import Path

# Carrega .env se existir
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

# Inicia
import uvicorn
uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
