import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional.text import char_error_rate
from torchmetrics.text import CharErrorRate


class TabulatureLightningModel(pl.LightningModule):
    def __init__(self, num_classes=29, hidden_size=256, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # Architektura CNN + RNN
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(input_size=256, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

        # INICJALIZACJA METRYKI CER
        self.val_cer = CharErrorRate()

    def decode_prediction(self, pred_indices):
        """ Zamienia listę ID (tokenów) na czytelny tekst string """
        # 1. Redukcja CTC (usuwanie powtórzeń i blanków '0')
        decoded_tokens = []
        previous_token = -1
        for token in pred_indices:
            if token != previous_token and token != 0:
                decoded_tokens.append(token)
            previous_token = token

        # 2. Mapowanie na znaki
        text = ""
        for t in decoded_tokens:
            if t == 26:
                text += ":"
            elif t == 27:
                text += ","
            elif t == 28:
                text += "|"
            else:
                text += str(int(t) - 1)
        return text

    def forward(self, x):
        x = self.cnn(x)
        x = self.pool(x)
        x = x.squeeze(2).permute(0, 2, 1)
        x, _ = self.rnn(x)
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

        # 1. Obliczanie Loss
        preds_for_loss = preds.permute(1, 0, 2)
        preds_log_softmax = nn.functional.log_softmax(preds_for_loss, dim=2)
        batch_size = preds.size(0)
        input_lengths = torch.full((batch_size,), preds.size(1), dtype=torch.long)
        loss = self.loss_fn(preds_log_softmax, targets, input_lengths, target_lengths)

        # 2. OBLICZANIE CER DLA CAŁEGO BATCHA
        # Wyciągamy najbardziej prawdopodobne znaki (Greedy Decoding)
        _, max_indices = torch.max(preds, dim=2)

        predicted_strings = []
        target_strings = []

        # Musimy "odpakować" targets (które są spłaszczone przez collate_fn)
        current_idx = 0
        for i in range(batch_size):
            # Przewidywanie maszyny
            p_str = self.decode_prediction(max_indices[i].cpu().numpy())
            predicted_strings.append(p_str)

            # Prawdziwa etykieta
            t_len = target_lengths[i].item()
            t_tokens = targets[current_idx: current_idx + t_len].cpu().numpy()
            t_str = self.decode_prediction(t_tokens)  # używamy dekodera, żeby formaty się zgadzały
            target_strings.append(t_str)
            current_idx += t_len

        # Aktualizujemy metrykę CER
        self.val_cer.update(predicted_strings, target_strings)

        # Logowanie wyników
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log('val_cer', self.val_cer, prog_bar=True, on_epoch=True, batch_size=batch_size)

        # Podgląd na żywo dla pierwszej paczki
        if batch_idx == 0:
            sample_cer = char_error_rate(predicted_strings[0], target_strings[0])
            self.print(f"\nEpoch {self.current_epoch} - WALIDACJA")
            self.print(f"Target: {target_strings[0]}")
            self.print(f"Model:  {predicted_strings[0]}")
            # self.print(f"Błąd CER: {sample_cer * 100:.1f}%")

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)