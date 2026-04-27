import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, random_split
from dataset import TabulatureDataset, collate_fn_ctc
from model import TabulatureLightningModel
from datetime import datetime





torch.set_float32_matmul_precision('medium')

DATA_DIR = '..\\dataset'

BATCH_SIZE = 20
NUM_WORKERS = 4 
MAX_EPOCHS = 500 
PATIENCE = 10





def main():
    print("\n[1/4] Inicjalizacja i podział Datasetu...")
    # Podajemy tylko zunifikowany DATA_DIR
    full_dataset = TabulatureDataset(data_dir=DATA_DIR)

    # podział na dane treningowe i walidacyjne
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f" -> Trening: {train_size} próbek | Walidacja: {val_size} próbek")

    # dataloadery
    print("[2/4] Konfiguracja Dataloaderów...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_ctc,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_ctc,
        num_workers=NUM_WORKERS,
        persistent_workers=True
    )

    # model
    print("[3/4] Inicjalizacja Modelu i Narzędzi...")
    model = TabulatureLightningModel(num_classes=29)
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # callbacki
    checkpoint_callback = ModelCheckpoint(
        dirpath='saved_models',
        filename=f'best_model_{start_time}',
        save_top_k=1,
        monitor='val_cer',
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_cer',
        patience=PATIENCE,
        verbose=True,
        mode='min'
    )

    # trainer
    print("[4/4] Start Treningu!\n")
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='auto',
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    # odpalenie maszyny
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()