import numpy as np
import os
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

### New imports for Lightning
import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, BackboneFinetuning
torch.set_float32_matmul_precision("medium")

# --- surgical additions (LitData streaming from S3) ---
import io
from PIL import Image
from litdata import StreamingDataset
# --- end additions ---

### Configure the training job
config = {
    "initial_epochs": 5,
    "total_epochs": 20,
    "patience": 5,
    "batch_size": 32,
    "lr": 1e-4,
    "fine_tune_lr": 1e-6,
    "model_architecture": "MobileNetV2",
    "dropout_probability": 0.5,
    "random_horizontal_flip": 0.5,
    "random_rotation": 15,
    "color_jitter_brightness": 0.2,
    "color_jitter_contrast": 0.2,
    "color_jitter_saturation": 0.2,
    "color_jitter_hue": 0.1,
}

### Prepare data loaders

train_transform = transforms.Compose(
    [
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_test_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# --- surgical additions (LitData S3 config) ---
S3_BUCKET = os.getenv("S3_BUCKET", "rb-litdata-food11")
LITDATA_PREFIX = os.getenv("LITDATA_PREFIX", "litdata_food11")
LITDATA_CACHE_DIR = os.getenv("LITDATA_CACHE_DIR", os.path.join(os.path.expanduser("~"), "litdata_cache"))
os.makedirs(LITDATA_CACHE_DIR, exist_ok=True)

TRAIN_URL = f"s3://{S3_BUCKET}/{LITDATA_PREFIX}/training"
VAL_URL = f"s3://{S3_BUCKET}/{LITDATA_PREFIX}/validation"
TEST_URL = f"s3://{S3_BUCKET}/{LITDATA_PREFIX}/evaluation"

S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

if not (S3_ENDPOINT_URL and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY):
    raise RuntimeError(
        "Missing S3 env vars. Set: S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY "
        "(and optionally AWS_DEFAULT_REGION)."
    )

STORAGE_OPTIONS = {"endpoint_url": S3_ENDPOINT_URL}
SESSION_OPTIONS = {
    "aws_access_key_id": AWS_ACCESS_KEY_ID,
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "region_name": AWS_DEFAULT_REGION,
}

DATALOADER_WORKERS = int(os.getenv("DATALOADER_WORKERS", "0"))
PERSISTENT_WORKERS = DATALOADER_WORKERS > 0
PREFETCH_FACTOR = int(os.getenv("PREFETCH_FACTOR", "2")) if DATALOADER_WORKERS > 0 else None
MAX_PRE_DOWNLOAD = int(os.getenv("LITDATA_MAX_PRE_DOWNLOAD", "16"))
# --- end additions ---


def decode_litdata_sample(sample, transform):
    img_bytes = sample["image"]
    label = int(sample["label"])
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if transform:
        image = transform(image)
    return image, label


class LitDataFood11(torch.utils.data.Dataset):
    def __init__(self, input_url: str, split_name: str, transform, shuffle: bool):
        self.transform = transform
        cache_dir = os.path.join(LITDATA_CACHE_DIR, split_name)
        os.makedirs(cache_dir, exist_ok=True)

        self.ds = StreamingDataset(
            input_dir=input_url,
            cache_dir=cache_dir,
            shuffle=shuffle,
            max_pre_download=MAX_PRE_DOWNLOAD,
            storage_options=STORAGE_OPTIONS,
            session_options=SESSION_OPTIONS,
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return decode_litdata_sample(self.ds[idx], self.transform)


train_dataset = LitDataFood11(TRAIN_URL, "training", train_transform, shuffle=True)
val_dataset = LitDataFood11(VAL_URL, "validation", val_test_transform, shuffle=False)
test_dataset = LitDataFood11(TEST_URL, "evaluation", val_test_transform, shuffle=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=DATALOADER_WORKERS,
    pin_memory=True,
    persistent_workers=PERSISTENT_WORKERS,
    **({} if PREFETCH_FACTOR is None else {"prefetch_factor": PREFETCH_FACTOR}),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=DATALOADER_WORKERS,
    pin_memory=True,
    persistent_workers=PERSISTENT_WORKERS,
    **({} if PREFETCH_FACTOR is None else {"prefetch_factor": PREFETCH_FACTOR}),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=DATALOADER_WORKERS,
    pin_memory=True,
    persistent_workers=PERSISTENT_WORKERS,
    **({} if PREFETCH_FACTOR is None else {"prefetch_factor": PREFETCH_FACTOR}),
)


class LightningFood11Model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")
        num_ftrs = self.model.last_channel
        self.model.classifier = nn.Sequential(nn.Dropout(config["dropout_probability"]), nn.Linear(num_ftrs, 11))
        self.criterion = nn.CrossEntropyLoss()

    @property
    def backbone(self):
        return self.model.features

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("train_accuracy", acc, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        return {"loss": loss, "train_accuracy": acc}

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("val_accuracy", acc, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_accuracy": acc}

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("test_loss", loss)
        self.log("test_accuracy", acc)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=config["lr"])
        return optimizer


checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
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

backbone_finetuning_callback = BackboneFinetuning(
    unfreeze_backbone_at_epoch=config["initial_epochs"],
    backbone_initial_lr=config["fine_tune_lr"],
    should_align=True,
)

lightning_food11_model = LightningFood11Model()

trainer = Trainer(
    max_epochs=config["total_epochs"],
    accelerator="gpu",
    devices="auto",
    num_sanity_val_steps=0,
    callbacks=[checkpoint_callback, early_stopping_callback, backbone_finetuning_callback],
)

trainer.fit(lightning_food11_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(lightning_food11_model, dataloaders=test_loader)