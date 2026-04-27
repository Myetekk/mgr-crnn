import os
from PIL import Image

DATA_DIR = '..\\dataset'  # Upewnij się, że ścieżka jest poprawna


def clean_dataset():
    print(f"Rozpoczynam skanowanie folderu: {DATA_DIR}")
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    corrupted_count = 0

    for img_name in files:
        img_path = os.path.join(DATA_DIR, img_name)
        try:
            # Próbujemy w pełni załadować obraz, żeby sprawdzić czy nie jest uszkodzony
            with Image.open(img_path) as img:
                img.load()
        except Exception as e:
            print(f"Zepsuty plik znaleziony: {img_name} - Usuwam!")
            # 1. Usuwamy zepsuty obrazek
            os.remove(img_path)

            # 2. Usuwamy odpowiadający mu plik .txt
            base_name = os.path.splitext(img_name)[0]
            label_path = os.path.join(DATA_DIR, f"{base_name}.txt")
            if os.path.exists(label_path):
                os.remove(label_path)

            corrupted_count += 1

    print(f"\nSkanowanie zakończone. Usunięto zepsutych par (obrazek+tekst): {corrupted_count}")


if __name__ == '__main__':
    clean_dataset()