import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
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
        num_workers=4,
        persistent_workers = True
    )

    # Taśmociąg do kartkówek (nie mieszamy - shuffle=False)
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn_ctc, num_workers=4
    )

    print("Inicjalizacja Modelu...")
    model = TabulatureLightningModel(num_classes=29)

    checkpoint_callback = ModelCheckpoint(
        dirpath='saved_models',
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,  # Zapisz tylko jeden, absolutnie najlepszy wynik
        monitor='val_loss',
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )

    print("Start treningu!")
    trainer = pl.Trainer(
        max_epochs=100, # Zwiększamy do 100!
        accelerator='auto',
        log_every_n_steps=10, # Możesz zmienić na 10 przy dużym zbiorze, żeby nie spamić logami
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, early_stop_callback] # <--- PODPINAMY GADŻETY
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()