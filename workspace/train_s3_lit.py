import numpy as np
import os
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from litdata.streaming import StreamingDataLoader
from torchvision import models, transforms

import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

torch.set_float32_matmul_precision("medium")

import io
from PIL import Image
from litdata import StreamingDataset

config = {
    "initial_epochs": 5,
    "total_epochs": 20,
    "patience": 5,
    "batch_size": int(os.getenv("BATCH_SIZE", "64")),
    "lr": 1e-4,
    "model_architecture": "MobileNetV2",
    "dropout_probability": 0.5,
    "random_horizontal_flip": 0.5,
    "random_rotation": 15,
    "color_jitter_brightness": 0.2,
    "color_jitter_contrast": 0.2,
    "color_jitter_saturation": 0.2,
    "color_jitter_hue": 0.1,
}

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=config["random_horizontal_flip"]),
    transforms.RandomRotation(config["random_rotation"]),
    transforms.ColorJitter(
        brightness=config["color_jitter_brightness"],
        contrast=config["color_jitter_contrast"],
        saturation=config["color_jitter_saturation"],
        hue=config["color_jitter_hue"],
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

S3_BUCKET = os.getenv("S3_BUCKET", "ansh-lab4-tests-bucket")
LITDATA_PREFIX = os.getenv("LITDATA_PREFIX", "optimized-dataset")
LITDATA_CACHE_DIR = os.getenv("LITDATA_CACHE_DIR", "./litdata_cache")
os.makedirs(LITDATA_CACHE_DIR, exist_ok=True)

TRAIN_URL = f"s3://{S3_BUCKET}/{LITDATA_PREFIX}/training"
VAL_URL = f"s3://{S3_BUCKET}/{LITDATA_PREFIX}/validation"
TEST_URL = f"s3://{S3_BUCKET}/{LITDATA_PREFIX}/evaluation"

STORAGE_OPTIONS = {"endpoint_url": os.environ["S3_ENDPOINT_URL"]}
SESSION_OPTIONS = {
    "aws_access_key_id": os.environ["ACCESS_KEY_ID"],
    "aws_secret_access_key": os.environ["SECRET_ACCESS_KEY"],
    # "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
}

DATALOADER_WORKERS = int(os.getenv("DATALOADER_WORKERS", "32"))
PREFETCH_FACTOR = int(os.getenv("PREFETCH_FACTOR", "128")) if DATALOADER_WORKERS > 0 else None
MAX_PRE_DOWNLOAD = int(os.getenv("LITDATA_MAX_PRE_DOWNLOAD", "128"))

def decode_litdata_sample(sample, transform):
    img = Image.open(io.BytesIO(sample["image"])).convert("RGB")
    if transform:
        img = transform(img)
    return img, int(sample["label"])

def make_collate(transform):
    def _collate(batch):
        xs, ys = zip(*(decode_litdata_sample(s, transform) for s in batch))
        return torch.stack(xs), torch.tensor(ys)
    return _collate

train_dataset = StreamingDataset(
    input_dir=TRAIN_URL,
    cache_dir=LITDATA_CACHE_DIR,
    shuffle=True,
    max_pre_download=MAX_PRE_DOWNLOAD,
    storage_options=STORAGE_OPTIONS,
    session_options=SESSION_OPTIONS,
)

val_dataset = StreamingDataset(
    input_dir=VAL_URL,
    cache_dir=LITDATA_CACHE_DIR,
    shuffle=False,
    max_pre_download=MAX_PRE_DOWNLOAD,
    storage_options=STORAGE_OPTIONS,
    session_options=SESSION_OPTIONS,
)

test_dataset = StreamingDataset(
    input_dir=TEST_URL,
    cache_dir=LITDATA_CACHE_DIR,
    shuffle=False,
    max_pre_download=MAX_PRE_DOWNLOAD,
    storage_options=STORAGE_OPTIONS,
    session_options=SESSION_OPTIONS,
)

train_loader = StreamingDataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    num_workers=DATALOADER_WORKERS,
    shuffle=True,
    collate_fn=make_collate(train_transform),
    **({} if PREFETCH_FACTOR is None else {"prefetch_factor": PREFETCH_FACTOR}),
)

val_loader = StreamingDataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    num_workers=DATALOADER_WORKERS,
    shuffle=False,
    collate_fn=make_collate(val_test_transform),
    **({} if PREFETCH_FACTOR is None else {"prefetch_factor": PREFETCH_FACTOR}),
)

test_loader = StreamingDataLoader(
    test_dataset,
    batch_size=config["batch_size"],
    num_workers=DATALOADER_WORKERS,
    shuffle=False,
    collate_fn=make_collate(val_test_transform),
    **({} if PREFETCH_FACTOR is None else {"prefetch_factor": PREFETCH_FACTOR}),
)

class LightningFood11Model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v2(
            weights="MobileNet_V2_Weights.DEFAULT"
        )

        for p in self.model.features.parameters():
            p.requires_grad = False

        num_ftrs = self.model.last_channel
        self.model.classifier = nn.Sequential(
            nn.Dropout(config["dropout_probability"]),
            nn.Linear(num_ftrs, 11),
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        acc = (out.argmax(1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        acc = (out.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, _):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        acc = (out.argmax(1) == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.classifier.parameters(), lr=config["lr"])

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="food11",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
)

early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    patience=config["patience"],
    mode="min",
)

model = LightningFood11Model()

trainer = Trainer(
    max_epochs=config["initial_epochs"],
    accelerator="gpu",
    devices="auto",
    num_sanity_val_steps=0,
    callbacks=[checkpoint_callback, early_stopping_callback],
)

trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)
