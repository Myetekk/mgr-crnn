import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
import os
import re





class TabulatureDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = sorted([
            f for f in os.listdir(data_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.transform = transforms.Compose([
            ResizeHeightOnly(target_h=64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.vocab = {}
        for i in range(25):
            self.vocab[f"digit.{i}"] = i + 1
        self.vocab["digit.X"] = 26
        self.vocab[':'] = 27
        self.vocab['+'] = 28
        self.vocab[' '] = 29
        for i in range(1, 7):
            self.vocab[str(i)] = 29 + i



    def __len__(self):
        return len(self.image_files)



    def tokenize(self, text):
        text = text.replace(' + ', '+')
        parts = re.findall(r'digit\.[0-9X]+|\d+|:|\+| ', text)
        return [self.vocab[p] for p in parts if p in self.vocab]



    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(self.data_dir, f"{base_name}_b.txt")

        with open(label_path, 'r') as f:
            text_label = f.read().strip()

        tokens = self.tokenize(text_label)
        return image, torch.tensor(tokens, dtype=torch.long)





class ResizeHeightOnly:
    def __init__(self, target_h=64):
        self.target_h = target_h

    def __call__(self, img):
        w, h = img.size
        new_w = max(1, int(w * (self.target_h / h)))
        return img.resize((new_w, self.target_h), Image.Resampling.BILINEAR)





def collate_fn_ctc(batch):
    images, targets = zip(*batch)
    max_w = max(img.shape[2] for img in images)

    val_r = (1.0 - 0.485) / 0.229
    val_g = (1.0 - 0.456) / 0.224
    val_b = (1.0 - 0.406) / 0.225

    padded_images = []
    for img in images:
        pad_w = max_w - img.shape[2]
        if pad_w > 0:
            pad_tensor = torch.zeros(3, img.shape[1], pad_w)
            pad_tensor[0, :, :] = val_r
            pad_tensor[1, :, :] = val_g
            pad_tensor[2, :, :] = val_b
            img = torch.cat([img, pad_tensor], dim=2)
        padded_images.append(img)

    images = torch.stack(padded_images, 0)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets = torch.cat(targets)

    return images, targets, target_lengths