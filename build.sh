#!/usr/bin/env bash
echo "🔧 Instalando dependencias..."
pip install -r requirements.txt

echo "⬇️ Descargando datasets y modelo..."
python scripts/download_data.py

echo "🚀 Iniciando aplicación..."
