import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional.text import char_error_rate
from torchmetrics.text import CharErrorRate





class TabulatureLightningModel(pl.LightningModule):
    def __init__(self, num_classes=29, hidden_size=256, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # Głęboka Architektura CNN (inspirowana VGG) z Batch Normalization
        self.cnn = nn.Sequential(
            # Blok 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Blok 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),  # Delikatny dropout wczesnych cech

            # Blok 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            # Blok 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(p=0.2),  # Mocniejszy dropout przed końcem CNN

            # Blok 5
            nn.Conv2d(512, 512, kernel_size=2, padding=0),
            nn.BatchNorm2d(512), nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d((1, None))  # spłaszczacz
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            batch_first=True
        )  # LongShortTermMemory - oczy z kontekstem
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # zgadywanie co widzi
        self.loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)  # ConnectionistTemporalClassification - nauczyciel

        self.val_cer = CharErrorRate()

    def decode_prediction(self, pred_indices):
        decoded_tokens = []
        previous_token = -1
        for token in pred_indices:
            if token != previous_token and token != 0:
                decoded_tokens.append(token)
            previous_token = token

        text = ""
        is_fret = True

        for t in decoded_tokens:
            if t == 26:
                text += ":"
                is_fret = False
            elif t == 27:
                text += " + "
                is_fret = True
            elif t == 28:
                text += " "
                is_fret = True
            else:
                num_str = str(int(t) - 1)

                if is_fret:
                    text += f"digit.{num_str}"
                else:
                    text += num_str

        return text



    def forward(self, x):
        x = self.cnn(x)
        x = self.pool(x)
        x = x.squeeze(2).permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        return self.fc(x)



    def training_step(self, batch, batch_idx):
        images, targets, target_lengths = batch
        preds = self(images).permute(1, 0, 2)
        preds_log_softmax = nn.functional.log_softmax(preds, dim=2)
        batch_size = preds.size(1)
        input_lengths = torch.full((batch_size,), preds.size(0), dtype=torch.long)
        loss = self.loss_fn(preds_log_softmax, targets, input_lengths, target_lengths)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        return loss



    def validation_step(self, batch, batch_idx):
        images, targets, target_lengths = batch
        preds = self(images)
        preds_for_loss = preds.permute(1, 0, 2)
        preds_log_softmax = nn.functional.log_softmax(preds_for_loss, dim=2)
        batch_size = preds.size(0)
        input_lengths = torch.full((batch_size,), preds.size(1), dtype=torch.long)
        loss = self.loss_fn(preds_log_softmax, targets, input_lengths, target_lengths)

        _, max_indices = torch.max(preds, dim=2)
        predicted_strings = []
        target_strings = []
        current_idx = 0
        for i in range(batch_size):
            p_str = self.decode_prediction(max_indices[i].cpu().numpy())
            predicted_strings.append(p_str)
            t_len = target_lengths[i].item()
            t_tokens = targets[current_idx: current_idx + t_len].cpu().numpy()
            t_str = self.decode_prediction(t_tokens)
            target_strings.append(t_str)
            current_idx += t_len

        self.val_cer.update(predicted_strings, target_strings)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log('val_cer', self.val_cer, prog_bar=True, on_epoch=True, batch_size=batch_size)

        if batch_idx == 0:
            self.print(f"\nEpoch {self.current_epoch} - WALIDACJA")
            self.print(f"Target: {target_strings[0]}")
            self.print(f"Model:  {predicted_strings[0]}")

        return loss



    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)

        # Zmniejszy Learning Rate o połowę, gdy walidacja (val_cer) nie poprawi się przez 8 epok
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=8,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_cer",
            },
        }