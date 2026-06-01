import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
import os
import re





class TabulatureDataset(Dataset):
    def __init__(self, data_dir):
        """ Inicjalizacja: podajemy tylko jeden folder, w którym są i obrazki i etykiety """
        self.data_dir = data_dir

        # Pobieramy TYLKO pliki graficzne, żeby nie próbować wczytać plików .txt jako obrazów!
        self.image_files = sorted([
            f for f in os.listdir(data_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        # zmieniamy obrazek na Tensor (macierz) i normalizujemy kolory
        self.transform = transforms.Compose([
            ResizeAndPad(64, 1024),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])



    def __len__(self):
        """ Zwraca ile łącznie mamy obrazków """
        return len(self.image_files)

    def tokenize(self, text):
        tokens = []
        text = text.replace('digit.', '')
        text = text.replace(' + ', '+')
        parts = re.findall(r'\d+|:|\+| ', text)

        for part in parts:
            if part == ':':
                tokens.append(26)
            elif part == '+':
                tokens.append(27)
            elif part == ' ':
                tokens.append(28)
            else:
                number = int(part)
                tokens.append(number + 1)

        return tokens



    def __getitem__(self, idx):
        """ Pobiera JEDEN konkretny obrazek i jego etykietę z dysku """
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)

        # Otwieramy obrazek i zmieniamy go na format zgodny z siecią (RGB)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # Szukamy odpowiadającego pliku txt w tym samym folderze.
        # Bezpiecznie odcinamy rozszerzenie obrazka i dodajemy .txt
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(self.data_dir, f"{base_name}_b.txt")

        with open(label_path, 'r') as f:
            text_label = f.read().strip()

        # Zamieniamy tekst na nasze tokeny
        tokens = self.tokenize(text_label)

        return image, torch.tensor(tokens, dtype=torch.long)





class ResizeAndPad:
    def __init__(self, target_h=64, target_w=1024):
        self.target_h = target_h
        self.target_w = target_w

    def __call__(self, img):
        # 1. Zmiana wysokości na 64 z zachowaniem proporcji (AspectRatio)
        w, h = img.size
        new_w = int(w * (self.target_h / h))
        img = img.resize((new_w, self.target_h), Image.Resampling.BILINEAR)

        # 2. Pad (dodawanie tła) lub Crop (ucinanie) do 1024
        if new_w < self.target_w:
            # Dodajemy białe tło (255, 255, 255) z prawej strony
            padding = (0, 0, self.target_w - new_w, 0)  # left, top, right, bottom
            img = TF.pad(img, padding, fill=(255, 255, 255))
        elif new_w > self.target_w:
            # Jeśli tabulatura jest za długa, brutalnie tniemy do 1024
            img = img.crop((0, 0, self.target_w, self.target_h))

        return img





def collate_fn_ctc(batch):
    images, targets = zip(*batch)
    # Obrazki pakujemy w jeden wielki tensor: [Rozmiar_Paczki, Kanały, Wys, Szer]
    images = torch.stack(images, 0)

    # Obliczamy jak długa jest każda etykieta w paczce (potrzebne dla nauczyciela)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)

    # Spłaszczamy wszystkie etykiety do jednej długiej dżdżownicy (wymóg CTCLoss w PyTorch)
    targets = torch.cat(targets)

    return images, targets, target_lengths