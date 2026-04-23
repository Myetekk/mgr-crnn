import torch
import os
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from torchvision import transforms
from model import TabulatureLightningModel
from torchmetrics.functional.text import char_error_rate





CHECKPOINT_PATH = 'saved_models/best_model_2026-04-22_15-37.ckpt'

DATA_DIR = '..\\testset'
RESULTS_DIR = '..\\results'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'





def test_model():
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = os.path.join(RESULTS_DIR, now)
    os.makedirs(output_dir, exist_ok=True)

    print(f"=========================================")
    print(f" ŁADOWANIE MODELU...")
    print(f"=========================================")
    print(f"Plik: {CHECKPOINT_PATH}")
    model = TabulatureLightningModel.load_from_checkpoint(CHECKPOINT_PATH)
    model.to(DEVICE)
    model.eval()

    images = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(images)

    print(f"\n=========================================")
    print(f" ROZPOCZĘCIE TESTÓW")
    print(f"=========================================")
    print(f"Ilość tabulatur w zbiorze testowym: {total_images}\n")

    if total_images == 0:
        print("Błąd: Nie znaleziono obrazów w folderze testowym!")
        return

    transform = transforms.Compose([
        transforms.Resize((64, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    total_accuracy = 0.0
    exact_matches = 0

    # Zmienne do zliczania pojedynczych znaków
    global_total_chars = 0
    global_correct_chars = 0

    with torch.no_grad():
        for img_name in images:
            # A. Wczytanie obrazu
            img_path = os.path.join(DATA_DIR, img_name)
            orig_img = Image.open(img_path).convert('RGB')
            input_tensor = transform(orig_img).unsqueeze(0).to(DEVICE)

            # B. Wczytanie prawdziwej etykiety
            label_path = os.path.join(DATA_DIR, img_name.replace('.png', '_b.txt'))
            true_text = "Brak pliku etykiety"
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    true_text = f.read().strip()

            # C. Predykcja modelu
            preds = model(input_tensor)
            _, max_indices = torch.max(preds, dim=2)
            decoded_text = model.decode_prediction(max_indices[0].cpu().numpy())

            # --- ZOSTAWIONE ZGODNIE Z PROŚBĄ ---
            decoded_text = decoded_text.replace('|', ' | ')

            # --- MATEMATYKA: Obliczanie skuteczności (CAR) i znaków ---
            if true_text != "Brak pliku etykiety":
                cer = char_error_rate(decoded_text, true_text).item()
                accuracy = max(0.0, 100.0 - (cer * 100))

                # Obliczanie pojedynczych znaków na podstawie błędu Levenshteina
                target_len = len(true_text)
                errors = int(round(cer * target_len))
                correct_chars = max(0, target_len - errors)

                global_total_chars += target_len
                global_correct_chars += correct_chars
            else:
                accuracy = 0.0

            total_accuracy += accuracy
            if accuracy == 100.0:
                exact_matches += 1

            # --- KONSOLA ---
            print(f"[{img_name}]")
            print(f"Prawda: {true_text}")
            print(f"Model:  {decoded_text}")
            print(f"Skuteczność: {accuracy:.1f}%\n")

            # =================================================================
            # D. Tworzenie obrazu wynikowego - KOLUMNY I WYRÓWNANIE
            # =================================================================
            try:
                font = ImageFont.truetype("arial.ttf", 22)
            except:
                font = ImageFont.load_default()

            dummy_img = Image.new('RGB', (1, 1))
            dummy_draw = ImageDraw.Draw(dummy_img)

            # Etykiety tekstowe na obrazie
            lbl_prawda = "Prawda: "
            lbl_model = "Model: "
            lbl_acc = "Skuteczność: "
            acc_str = f"{accuracy:.1f}%"

            # Funkcja pomocnicza do mierzenia szerokości tekstu
            def text_width(text):
                return dummy_draw.textbbox((0, 0), text, font=font)[2]

            # Mierzymy najdłuższą etykietę i najdłuższą wartość
            max_lbl_w = max(text_width(lbl_prawda), text_width(lbl_model), text_width(lbl_acc))
            max_val_w = max(text_width(true_text), text_width(decoded_text), text_width(acc_str))

            total_text_width = max_lbl_w + max_val_w
            margin = 40
            new_width = max(orig_img.width, int(total_text_width) + margin)

            footer_height = 120
            new_height = orig_img.height + footer_height

            combined_img = Image.new('RGB', (new_width, new_height), (245, 245, 245))

            # Centrujemy oryginalną tabulaturę na górze
            img_x_offset = (new_width - orig_img.width) // 2
            combined_img.paste(orig_img, (img_x_offset, 0))

            # Rysowanie tekstu - osie kolumn
            draw = ImageDraw.Draw(combined_img)
            start_x = (new_width - total_text_width) // 2
            val_x = start_x + max_lbl_w

            # 1 linia: Prawda
            draw.text((start_x, orig_img.height + 15), lbl_prawda, fill=(50, 50, 50), font=font)
            draw.text((val_x, orig_img.height + 15), true_text, fill=(50, 50, 50), font=font)

            # 2 linia: Model
            draw.text((start_x, orig_img.height + 45), lbl_model, fill=(0, 100, 0), font=font)
            draw.text((val_x, orig_img.height + 45), decoded_text, fill=(0, 100, 0), font=font)

            # 3 linia: Skuteczność
            acc_color = (0, 150, 0) if accuracy == 100.0 else (200, 0, 0)
            draw.text((start_x, orig_img.height + 75), lbl_acc, fill=(50, 50, 50), font=font)
            draw.text((val_x, orig_img.height + 75), acc_str, fill=acc_color, font=font)

            # E. Zapis
            combined_img.save(os.path.join(output_dir, f"cmp_{img_name}"))

    # =================================================================
    # PODSUMOWANIE STATYSTYK I ZAPIS DO PLIKU
    # =================================================================
    avg_accuracy = total_accuracy / total_images if total_images > 0 else 0.0
    char_accuracy_pct = (global_correct_chars / global_total_chars * 100) if global_total_chars > 0 else 0.0

    # Formatowanie stringa z podsumowaniem
    summary_text = (
        f"========================================================\n"
        f" PODSUMOWANIE ZBIORU TESTOWEGO\n"
        f"========================================================\n"
        f" Przetestowano obrazów:  {total_images}\n"
        f" Idealne dopasowania:    {exact_matches} / {total_images} ({(exact_matches / total_images) * 100:.1f}%)\n"
        f" Rozpoznane znaki:       {global_correct_chars} / {global_total_chars} ({char_accuracy_pct:.3f}%)\n"
        f" Średnia skuteczność:    {avg_accuracy:.3f}%\n"
        f"========================================================\n"
    )

    # Wypisanie w konsoli
    print(summary_text)
    print(f"Zakończono! Porównania graficzne znajdziesz w:\n{output_dir}")

    # Zapisanie do pliku _result.txt w tym samym folderze
    result_file_path = os.path.join(output_dir, "_result.txt")
    with open(result_file_path, "w", encoding="utf-8") as f:
        f.write(summary_text)


if __name__ == '__main__':
    test_model()