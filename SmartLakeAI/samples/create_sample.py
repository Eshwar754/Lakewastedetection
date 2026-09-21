import shutil
from pathlib import Path

def copy_sample():
    src_dir = Path(r"c:\Users\smesh\Downloads\trash_inst_material\trash_inst_material\val\images")
    dest_dir = Path(__file__).resolve().parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    images = list(src_dir.glob("*.*"))
    if images:
        shutil.copy(images[0], dest_dir / "sample_lake_trash.jpg")
        print(f"[+] Sample image copied to: {dest_dir / 'sample_lake_trash.jpg'}")

if __name__ == '__main__':
    copy_sample()
