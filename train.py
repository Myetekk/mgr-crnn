import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, random_split
from dataset import TabulatureDataset, collate_fn_ctc
from model import TabulatureLightningModel
from datetime import datetime
import os
import glob





torch.set_float32_matmul_precision('medium')

# MODEL_VERSION = 'noise_MAX'
# MODEL_VERSION = 'noise_MID'
# MODEL_VERSION = 'noise_MIN'
# MODEL_VERSION = 'add_MAX'
# MODEL_VERSION = 'add_MID'
# MODEL_VERSION = 'add_MIN'
# MODEL_VERSION = 'time_MAX'
# MODEL_VERSION = 'time_MID'
# MODEL_VERSION = 'time_MIN'
MODEL_VERSION = 'all_MAX'
# MODEL_VERSION = 'all_MID'
# MODEL_VERSION = 'all_MIN'
DATA_DIR = f"..\\train_set\\{MODEL_VERSION}"

CHECKPOINT_BASENAME = f"model_{MODEL_VERSION}"
CHECKPOINT_NAME = f"{CHECKPOINT_BASENAME}.ckpt"
CHECKPOINT_DIR = 'saved_models'

BATCH_SIZE = 20
NUM_WORKERS = 4
MAX_EPOCHS = 500
PATIENCE = 20





def main():
    print("\n[1/5] Inicjalizacja i podział Datasetu...")
    full_dataset = TabulatureDataset(data_dir=DATA_DIR)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f" -> Trening: {train_size} próbek | Walidacja: {val_size} próbek")

    print("[2/5] Konfiguracja Dataloaderów...")
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

    print("[3/5] Inicjalizacja Modelu i Narzędzi...")
    model = TabulatureLightningModel(num_classes=36)

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename=CHECKPOINT_BASENAME,
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

    print("[4/5] Start Treningu!\n")
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='auto',
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    latest_ckpt = find_latest_checkpoint(CHECKPOINT_DIR)
    try:
        if latest_ckpt:
            print(f"\n[5/5] Znaleziono punkt kontrolny: {latest_ckpt}")
            print("      Wznawiam trening od miejsca przerwania...\n")
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=latest_ckpt)
        else:
            print(f"\n[5/5] Brak punktów kontrolnych w '{CHECKPOINT_DIR}'.")
            print("      Start Treningu od zera!\n")
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # model_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            end_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
            new_model_name = f"model_{end_time}_{MODEL_VERSION}.ckpt"
            new_model_path = os.path.join(CHECKPOINT_DIR, new_model_name)

            os.rename(best_model_path, new_model_path)
            print(f"\n[ZAKOŃCZONO SUKCESEM] Trening dobiegł końca!")
            print(f"Model został zarchiwizowany jako: {new_model_name}\n")

    except KeyboardInterrupt:
        print("\n\n[PRZERWANO] Trening zatrzymany ręcznie przez użytkownika.")
        print(f"Plik '{CHECKPOINT_NAME}' czeka w folderze na wznowienie treningu.\n")



def find_latest_checkpoint(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        return None
    list_of_files = glob.glob(f'{ckpt_dir}/{CHECKPOINT_NAME}')
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file



if __name__ == '__main__':
    main()