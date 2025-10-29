import os
import requests
from zipfile import ZipFile

# === CONFIGURACIÓN ===
# URLs de tus carpetas en Drive convertidas a descargas ZIP
folders = {
    "user_datasets": "https://drive.google.com/file/d/1SQD2lpRCVEF-hBOnLXbeiwdTS0XIyTTV/view?usp=drive_link",
    "datasets": "https://drive.google.com/file/d/1qV0FIrHDwlsgv4thV9sa5YB6rkPhklTN/view?usp=drive_link",
    "modelo": "https://drive.google.com/file/d/11u2pm9kWffxqdeaHQVCwI95vVudVl_1n/view?usp=drive_link",
}

# === FUNCIONES ===
def download_file(url, output):
    print(f"⬇️  Descargando {output} ...")
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        print(f"⚠️ Error descargando {url}")
        return
    with open(output, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ Descargado {output}")

def extract_zip(path, target_dir):
    print(f"📦 Extrayendo {path} ...")
    with ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    os.remove(path)
    print(f"✅ Extraído en {target_dir}")

# === DESCARGA Y EXTRACCIÓN ===
os.makedirs("data_cache", exist_ok=True)
for name, url in folders.items():
    target_zip = f"data_cache/{name}.zip"
    download_file(url, target_zip)
    extract_zip(target_zip, name)
