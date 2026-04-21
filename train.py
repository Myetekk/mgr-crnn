import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from dataset import TabulatureDataset, collate_fn_ctc
from model import TabulatureLightningModel
import torch
torch.set_float32_matmul_precision('medium')


def main():
    print("Inicjalizacja Datasetu...")
    # Ładujemy WSZYSTKIE wygenerowane dane
    full_dataset = TabulatureDataset(image_dir='..\\dataset\\images', label_dir='..\\dataset\\labels_model_b')

    # Dzielimy dane: 80% trening (nauka), 20% walidacja (kartkówka)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Dane podzielone: {train_size} do treningu, {val_size} do walidacji.")

    # Taśmociąg do nauki (mieszamy dane - shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=20,
        shuffle=True,
        collate_fn=collate_fn_ctc,
        num_workers=4
    )

    # Taśmociąg do kartkówek (nie mieszamy - shuffle=False)
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn_ctc, num_workers=4
    )

    print("Inicjalizacja Modelu...")
    model = TabulatureLightningModel(num_classes=29)

    print("Start treningu!")
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='auto',
        log_every_n_steps=1,
        check_val_every_n_epoch=1  # Robimy kartkówkę co 1 epokę
    )

    # Odpalamy trening podając OBA taśmociągi
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()