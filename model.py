import torch
import torch.nn as nn
import pytorch_lightning as pl


class TabulatureLightningModel(pl.LightningModule):
    def __init__(self, num_classes=29, hidden_size=256, learning_rate=1e-4):
        super().__init__()
        # Zapisujemy parametry, żeby PyTorch Lightning o nich pamiętał
        self.save_hyperparameters()

        # ============================================================
        # 1. OCZY MASZYNY (Własna sieć CNN)
        # ============================================================
        # Rezygnujemy z ResNeta na rzecz warstw, które lepiej czytają tekst.
        # Każda warstwa Conv2d szuka wzorców (linii, cyfr).
        # MaxPool2d zmniejsza obraz, żeby maszyna nie musiała liczyć każdego piksela.

        self.cnn = nn.Sequential(
            # Warstwa 1: Obraz 1024x64 -> 512x32 (zmniejszamy o połowę)
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Warstwa 2: 512x32 -> 256x16 (znowu o połowę)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Warstwa 3: 256x16 -> 256x8
            # UWAGA: Tutaj (kernel=(2,1)) zgniatamy TYLKO wysokość.
            # Szerokość zostaje, żeby nie zgubić gęsto zapisanych nut!
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            # Warstwa 4: 256x8 -> 256x4 (znowu tylko wysokość w dół)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        # Ostatnia zgniatarka: upewnia się, że wysokość to dokładnie 1.
        self.pool = nn.AdaptiveAvgPool2d((1, None))

        # ============================================================
        # 2. CZYTACZ (RNN - LSTM)
        # ============================================================
        # input_size=256, bo ostatnia warstwa CNN wypluwa 256 kanałów cech.
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )

        # Klasyfikator: decyduje, który to znak z naszego słownika (29 klas)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

        # Nauczyciel (CTC Loss). Blank=0 to nasz znak pusty.
        self.loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, x):
        # 1. Przepuszczamy obraz przez oczy (CNN)
        x = self.cnn(x)  # Wynik: [Batch, 256, 4, 256]
        x = self.pool(x)  # Wynik: [Batch, 256, 1, 256]

        # 2. Przygotowujemy dane do czytania (wywalamy wysokość, zamieniamy osie)
        x = x.squeeze(2)  # Wynik: [Batch, 256, 256]
        x = x.permute(0, 2, 1)  # Wynik: [Batch, 256, 256] (Teraz oś czasu jest na środku)

        # 3. Czytamy sekwencję
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        images, targets, target_lengths = batch

        # Maszyna zgaduje
        preds = self(images)

        # CTC wymaga, żeby czas był na pierwszym miejscu (wymóg techniczny PyTorcha)
        preds = preds.permute(1, 0, 2)

        # Logarytmowanie wyników dla Nauczyciela
        preds_log_softmax = nn.functional.log_softmax(preds, dim=2)

        # Sprawdzamy, jak długą kliszę wygenerowała maszyna
        batch_size = preds.size(1)
        time_steps = preds.size(0)
        input_lengths = torch.full(size=(batch_size,), fill_value=time_steps, dtype=torch.long)

        # Nauczyciel liczy błąd
        loss = self.loss_fn(preds_log_softmax, targets, input_lengths, target_lengths)

        # Zapisujemy wynik do paska postępu
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets, target_lengths = batch

        # 1. Maszyna zgaduje (na danych, których nie widziała podczas treningu)
        preds = self(images)

        # 2. Obliczamy błąd walidacyjny (val_loss) w ten sam sposób co treningowy
        preds_for_loss = preds.permute(1, 0, 2)
        preds_log_softmax = nn.functional.log_softmax(preds_for_loss, dim=2)

        batch_size = preds.size(0)
        time_steps = preds.size(1)
        input_lengths = torch.full(size=(batch_size,), fill_value=time_steps, dtype=torch.long)

        loss = self.loss_fn(preds_log_softmax, targets, input_lengths, target_lengths)

        # Zapisujemy val_loss (będzie się pojawiał na końcu każdej epoki)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)

        # =========================================================
        # MAGIA INŻYNIERSKA: Dekodowanie na żywo pierwszej próbki z batcha!
        # =========================================================
        if batch_idx == 0:  # Drukujemy tylko dla pierwszej paczki w epoce, żeby nie zalać konsoli
            # Bierzemy przewidywania dla pierwszego obrazka
            first_pred = preds[0]  # Kształt: [256 (czas), 29 (klasy)]

            # Szukamy, która klasa ma najwyższe prawdopodobieństwo w każdym kroku czasu
            _, max_indices = torch.max(first_pred, dim=1)
            predicted_tokens = max_indices.cpu().numpy()

            # Redukcja CTC (Usuwamy powtórzenia i znaki puste '0')
            decoded_tokens = []
            previous_token = -1
            for token in predicted_tokens:
                if token != previous_token and token != 0:
                    decoded_tokens.append(token)
                previous_token = token

            # Zamieniamy tokeny z powrotem na tekst
            decoded_text = ""
            for t in decoded_tokens:
                if t == 26:
                    decoded_text += ":"
                elif t == 27:
                    decoded_text += ","
                elif t == 28:
                    decoded_text += " | "
                else:
                    decoded_text += str(t - 1)  # Cofamy nasze +1 dla progów

            self.print(f"\n[EPOKA] Walidacja na żywo:")
            self.print(f"Model przeczytał: {decoded_text}")

        return loss

    def configure_optimizers(self):
        # Używamy Adama - to najpopularniejszy "poprawiacz" błędów
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)