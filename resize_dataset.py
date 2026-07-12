from PIL import Image
from pathlib import Path

root = Path('dataset')
for class_dir in root.iterdir():
    if not class_dir.is_dir():
        continue
    for img_path in class_dir.iterdir():
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff','avif','webp'}:
            continue
        with Image.open(img_path) as img:
            rgb_img = img.convert('RGB')
            resized = rgb_img.resize((224, 224), Image.Resampling.LANCZOS)
            resized.save(img_path, format='JPEG', quality=95)
            print(f'{img_path} -> {resized.size}')
