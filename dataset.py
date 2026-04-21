import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import re





class TabulatureDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        """ Inicjalizacja: podajemy ścieżki do folderów z danymi """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted(os.listdir(image_dir))

        # Transformacje: zmieniamy obrazek na Tensor (macierz) i normalizujemy kolory
        self.transform = transforms.Compose([
            transforms.Resize((64, 1024)),  # <--- POSZERZAMY OBRAZ!
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        """ Zwraca ile łącznie mamy obrazków """
        return len(self.image_files)

    def tokenize(self, text):
        tokens = []

        # re.findall to potężne narzędzie. Mówi: "Znajdź mi ciągi cyfr (\d+) LUB
        # dwukropek (:) LUB kreskę (\|) LUB przecinek (,). Zignoruj wszystko inne (np. spacje)".
        parts = re.findall(r'\d+|:|\||,', text)

        for part in parts:
            if part == ':':
                tokens.append(26)  # Zmienione ID (zostawiamy miejsce na progi)
            elif part == ',':
                tokens.append(27)
            elif part == '|':
                tokens.append(28)  # Nasz nowy separator
            else:
                # part to liczba. Dodajemy +1, aby ominąć token 0 [BLANK]
                # Próg 0 = klasa 1. Struna 6 = klasa 7. Próg 24 = klasa 25.
                number = int(part)
                tokens.append(number + 1)

        return tokens

    def __getitem__(self, idx):
        """ Pobiera JEDEN konkretny obrazek i jego etykietę z dysku """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Otwieramy obrazek i zmieniamy go na format zgodny z siecią (RGB)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # Szukamy odpowiadającego pliku txt.
        label_path = os.path.join(self.label_dir, img_name.replace('.png', '.txt'))
        with open(label_path, 'r') as f:
            text_label = f.read().strip()

        # Zamieniamy tekst na nasze tokeny
        tokens = self.tokenize(text_label)

        return image, torch.tensor(tokens, dtype=torch.long)





# ---------------------------------------------------------
# UWAGA: To jest kluczowa funkcja (HACZYK dla CTC Loss).
# Skleja nam wiele obrazków w jeden "Batch" (paczkę).
# ---------------------------------------------------------
def collate_fn_ctc(batch):
    images, targets = zip(*batch)

    # Obrazki pakujemy w jeden wielki tensor: [Rozmiar_Paczki, Kanały, Wys, Szer]
    images = torch.stack(images, 0)

    # Obliczamy jak długa jest każda etykieta w paczce (potrzebne dla nauczyciela)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)

    # Spłaszczamy wszystkie etykiety do jednej długiej dżdżownicy (wymóg CTCLoss w PyTorch)
    targets = torch.cat(targets)

    return images, targets, target_lengths