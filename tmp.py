import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import TabulatureDataset, collate_fn_ctc
from model import TabulatureLightningModel
import torch
torch.set_float32_matmul_precision('medium')


def main():
    print("Inicjalizacja Datasetu...")
    # Tutaj podajesz ścieżki do swoich wygenerowanych obrazków
    train_dataset = TabulatureDataset(image_dir='..\\dataset\\images', label_dir='..\\dataset\\labels_model_b')

    # Dataloader to taśmociąg, który będzie podawał obrazki po 16 sztuk na raz (batch_size)
    # collate_fn=collate_fn_ctc to wpięcie naszego haczyka z dataset.py
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,  # Mieszamy obrazki, żeby maszyna nie uczyła się na pamięć
        collate_fn=collate_fn_ctc,
        num_workers=4  # Używa kilku rdzeni procesora do wczytywania obrazków z dysku
    )

    print("Inicjalizacja Modelu...")
    # 27 klas (0=Blank, 1-24=Progi, 25=:, 26=,)
    model = TabulatureLightningModel(num_classes=29)

    print("Start treningu!")
    # Inicjalizujemy kierownika z PyTorch Lightning.
    # accelerator='auto' sprawi, że program sam poszuka karty graficznej (CUDA),
    # a jak nie znajdzie, użyje procesora (CPU).
    trainer = pl.Trainer(
        max_epochs=10,  # Przerobi cały Twój zbiór obrazków 50 razy
        accelerator='auto'
    )

    trainer.fit(model, train_dataloaders=train_loader)


if __name__ == '__main__':
    main()