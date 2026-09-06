import torch
import os
import re
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from torchvision import transforms
from model import TabulatureLightningModel
from torchmetrics.functional.text import char_error_rate, word_error_rate





CHECKPOINT_PATH = 'saved_models/model_2026-07-07-19-07.ckpt'

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
        ResizeHeightOnly(target_h=64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    total_accuracy = 0.0
    exact_matches = 0
    global_total_chars = 0
    global_correct_chars = 0
    global_total_words = 0
    global_correct_words = 0

    MAX_ORIGINAL_WIDTH = 1000

    with torch.no_grad():
        for img_name in images:
            img_path = os.path.join(DATA_DIR, img_name)
            orig_img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = orig_img.size

            label_path = os.path.join(DATA_DIR, img_name.replace('.png', '_b.txt'))
            label_path = label_path.replace('.jpg', '_b.txt').replace('.jpeg', '_b.txt')

            true_text = "Brak pliku etykiety"
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    true_text = f.read().strip()

            decoded_parts = []

            # 1. LOGIKA CIĘCIA NA ORYGINALNYM OBRAZIE (Przed transformacją)
            if orig_w > MAX_ORIGINAL_WIDTH:
                num_chunks = int((orig_w + MAX_ORIGINAL_WIDTH - 1) // MAX_ORIGINAL_WIDTH)

                for i in range(num_chunks):
                    # Wycinanie fizycznego fragmentu obrazka
                    left = i * MAX_ORIGINAL_WIDTH
                    right = min((i + 1) * MAX_ORIGINAL_WIDTH, orig_w)
                    chunk_img = orig_img.crop((left, 0, right, orig_h))

                    # Ensure minimum width to prevent CNN kernel crash
                    if chunk_img.width < 100:
                        safe_chunk = Image.new('RGB', (100, orig_h), (255, 255, 255))
                        safe_chunk.paste(chunk_img, (0, 0))
                        chunk_img = safe_chunk

                    # 2. Transformacja wycinka
                    chunk_tensor = transform(chunk_img).unsqueeze(0).to(DEVICE)

                    # 3. Predykcja
                    preds = model(chunk_tensor)
                    _, max_indices = torch.max(preds, dim=2)
                    chunk_text = model.decode_prediction(max_indices[0].cpu().numpy())

                    if chunk_text.strip():
                        decoded_parts.append(chunk_text.strip())

                # Sklejanie wyników z zabezpieczeniem przed podwójnymi plusami
                decoded_text = " + ".join(decoded_parts)
                decoded_text = re.sub(r'(\s*\+\s*)+', ' + ', decoded_text).strip(' +')

            else:
                # Jeśli obrazek jest wąski, procesujemy całość normalnie
                if orig_w < 100:
                    safe_img = Image.new('RGB', (100, orig_h), (255, 255, 255))
                    safe_img.paste(orig_img, (0, 0))
                    orig_img = safe_img

                input_tensor = transform(orig_img).unsqueeze(0).to(DEVICE)
                preds = model(input_tensor)
                _, max_indices = torch.max(preds, dim=2)
                decoded_text = model.decode_prediction(max_indices[0].cpu().numpy())

            # Usuwanie prefiksu do ładniejszego wyświetlania i liczenia CER
            decoded_text = decoded_text.replace('digit.', '')
            decoded_text = decoded_text.replace(' + ', '+')
            true_text = true_text.replace('digit.', '')
            true_text = true_text.replace(' + ', '+')

            if true_text != "Brak pliku etykiety":
                cer = char_error_rate(decoded_text, true_text).item()
                cer_accuracy = max(0.0, 100.0 - (cer * 100))

                target_len = len(true_text)
                errors = int(round(cer * target_len))
                correct_chars = max(0, target_len - errors)

                global_total_chars += target_len
                global_correct_chars += correct_chars

                decoded_text = decoded_text.replace('+', ' + ')
                true_text = true_text.replace('+', ' + ')
                
                wer = word_error_rate(decoded_text, true_text).item()
                target_words_len = len(true_text.split())
                errors_words = int(round(wer * target_words_len))
                correct_words = max(0, target_words_len - errors_words)
                
                global_total_words += target_words_len
                global_correct_words += correct_words
                wer_accuracy = max(0.0, 100.0 - (wer * 100))
            else:
                cer_accuracy = 0.0
                wer_accuracy = 0.0

            total_accuracy += cer_accuracy
            if cer_accuracy == 100.0:
                exact_matches += 1

            print(f"[{img_name}]")
            if orig_w > MAX_ORIGINAL_WIDTH:
                print(f"Szerokość oryginalna: {orig_w} px (podzielono na {num_chunks} części po max {MAX_ORIGINAL_WIDTH}px)")
            else:
                print(f"Szerokość oryginalna: {orig_w} px")

            print(f"Prawda: {true_text}")
            print(f"Model:  {decoded_text}")
            print(f"Skuteczność: {cer_accuracy:.1f}%\n")

            # --- Generowanie obrazka z wynikami ---
            try:
                font = ImageFont.truetype("arial.ttf", 22)
            except:
                font = ImageFont.load_default()

            dummy_img = Image.new('RGB', (1, 1))
            dummy_draw = ImageDraw.Draw(dummy_img)

            lbl_prawda = "Prawda: "
            lbl_model = "Model: "
            lbl_cer = "CER - znaki: "
            cer_str = f"{cer_accuracy:.1f}%"
            lbl_wer = "WER - tokeny: "
            wer_str = f"{wer_accuracy:.1f}%"

            def text_width(text):
                return dummy_draw.textbbox((0, 0), text, font=font)[2]

            max_lbl_w = max(text_width(lbl_prawda), text_width(lbl_model), text_width(lbl_cer), text_width(lbl_wer))
            max_val_w = max(text_width(true_text), text_width(decoded_text), text_width(cer_str), text_width(wer_str))

            total_text_width = max_lbl_w + max_val_w
            margin = 40
            new_width = max(orig_img.width, int(total_text_width) + margin)

            footer_height = 150
            new_height = orig_img.height + footer_height

            combined_img = Image.new('RGB', (new_width, new_height), (245, 245, 245))

            img_x_offset = (new_width - orig_img.width) // 2
            combined_img.paste(orig_img, (img_x_offset, 0))

            draw = ImageDraw.Draw(combined_img)
            start_x = (new_width - total_text_width) // 2
            val_x = start_x + max_lbl_w

            draw.text((start_x, orig_img.height + 15), lbl_prawda, fill=(50, 50, 50), font=font)
            draw.text((val_x, orig_img.height + 15), true_text, fill=(50, 50, 50), font=font)

            draw.text((start_x, orig_img.height + 45), lbl_model, fill=(0, 100, 0), font=font)
            draw.text((val_x, orig_img.height + 45), decoded_text, fill=(0, 100, 0), font=font)

            def get_color(acc):
                p = max(0.0, min(100.0, acc)) / 100.0
                # Interpolacja: od czerwieni (200, 0, 0) do zieleni (0, 160, 0)
                # Opcjonalnie przejście przez żółty: 
                # r = int(200 * min(1.0, 2.0 * (1.0 - p)))
                # g = int(160 * min(1.0, 2.0 * p))
                r = int(200 * min(1.0, 2.0 * (1.0 - p)))
                g = int(160 * min(1.0, 2.0 * p))
                return (r, g, 0)

            acc_color = get_color(cer_accuracy)
            draw.text((start_x, orig_img.height + 75), lbl_cer, fill=(50, 50, 50), font=font)
            draw.text((val_x, orig_img.height + 75), cer_str, fill=acc_color, font=font)

            wer_color = get_color(wer_accuracy)
            draw.text((start_x, orig_img.height + 105), lbl_wer, fill=(50, 50, 50), font=font)
            draw.text((val_x, orig_img.height + 105), wer_str, fill=wer_color, font=font)

            combined_img.save(os.path.join(output_dir, f"cmp_{img_name}"))

    avg_accuracy = total_accuracy / total_images if total_images > 0 else 0.0
    char_accuracy_pct = (global_correct_chars / global_total_chars * 100) if global_total_chars > 0 else 0.0
    word_accuracy_pct = (global_correct_words / global_total_words * 100) if global_total_words > 0 else 0.0

    summary_text = (
        f"========================================================\n"
        f" PODSUMOWANIE ZBIORU TESTOWEGO\n"
        f"========================================================\n"
        f" Przetestowano obrazów:   {total_images}\n"
        f" Idealne dopasowania:     {exact_matches} / {total_images} ({(exact_matches / total_images) * 100:.1f}%)\n"
        f" Rozpoznane znaki (CER):  {global_correct_chars} / {global_total_chars} ({char_accuracy_pct:.3f}%)\n"
        f" Rozpoznane tokeny (WER): {global_correct_words} / {global_total_words} ({word_accuracy_pct:.3f}%)\n"
        f" Średnia skuteczność:     {avg_accuracy:.3f}%\n"
        f"========================================================\n"
    )

    print(summary_text)
    result_file_path = os.path.join(output_dir, "_result.txt")
    with open(result_file_path, "w", encoding="utf-8") as f:
        f.write(summary_text)


class ResizeHeightOnly:
    def __init__(self, target_h=64):
        self.target_h = target_h

    def __call__(self, img):
        w, h = img.size
        new_w = max(1, int(w * (self.target_h / h)))
        return img.resize((new_w, self.target_h), Image.Resampling.BILINEAR)





if __name__ == '__main__':
    test_model()