import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, random_split
from dataset import TabulatureDataset, collate_fn_ctc
from model import TabulatureLightningModel
from datetime import datetime

# Optymalizacja dla kart NVIDIA RTX
torch.set_float32_matmul_precision('medium')

# =====================================================================
# USTAWIENIA GŁÓWNE (Hiperparametry)
# =====================================================================
DATA_IMAGES = '..\\dataset\\images'
DATA_LABELS = '..\\dataset\\labels_model_b'

BATCH_SIZE = 20       # Uwaga: zmniejsz do 16, jeśli wywali błąd CUDA out of memory
NUM_WORKERS = 4       # Uwaga: zmniejsz do 2, jeśli wywali błąd openBLAS (brak RAM)
MAX_EPOCHS = 100      # Maksymalna liczba epok treningu
PATIENCE = 10         # Early Stopping: po ilu epokach bez poprawy przerwać
# =====================================================================

def main():
    # --- 1. PRZYGOTOWANIE DANYCH ---
    print("\n[1/4] Inicjalizacja i podział Datasetu...")
    full_dataset = TabulatureDataset(image_dir=DATA_IMAGES, label_dir=DATA_LABELS)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f" -> Trening: {train_size} próbek | Walidacja: {val_size} próbek")

    # --- 2. TAŚMOCIĄGI (DATALOADERY) ---
    print("[2/4] Konfiguracja Dataloaderów...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_ctc,
        num_workers=NUM_WORKERS,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_ctc,
        num_workers=NUM_WORKERS,
        persistent_workers=True
    )

    # --- 3. MODEL I NARZĘDZIA (CALLBACKI) ---
    print("[3/4] Inicjalizacja Modelu i Narzędzi...")
    model = TabulatureLightningModel(num_classes=29)
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M")

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

    # --- 4. SILNIK TRENINGOWY (TRAINER) ---
    print("[4/4] Start Treningu!\n")
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='auto',
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    # Odpalenie maszyny
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()