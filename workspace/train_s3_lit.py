import os
import io
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import models, transforms

import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from litdata import StreamingDataset
from litdata.streaming import StreamingDataLoader

torch.set_float32_matmul_precision("medium")


config = {
    "total_epochs": 20,  
    "patience": 5,
    "batch_size": int(os.getenv("BATCH_SIZE", "32")),
    "lr": 1e-4,
    "dropout_probability": 0.5,
}


train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


S3_BUCKET = os.getenv("S3_BUCKET", "rb-litdata-food11")
LITDATA_PREFIX = os.getenv("LITDATA_PREFIX", "litdata_food11")
LITDATA_CACHE_DIR = os.getenv("LITDATA_CACHE_DIR", "/tmp/litdata_cache")

TRAIN_URL = f"s3://{S3_BUCKET}/{LITDATA_PREFIX}/training"
VAL_URL   = f"s3://{S3_BUCKET}/{LITDATA_PREFIX}/validation"
TEST_URL  = f"s3://{S3_BUCKET}/{LITDATA_PREFIX}/evaluation"

os.makedirs(LITDATA_CACHE_DIR, exist_ok=True)

DATALOADER_WORKERS = int(os.getenv("DATALOADER_WORKERS", "0"))
PREFETCH_FACTOR = int(os.getenv("PREFETCH_FACTOR", "2")) if DATALOADER_WORKERS > 0 else None
MAX_PRE_DOWNLOAD = int(os.getenv("LITDATA_MAX_PRE_DOWNLOAD", "16"))


def decode_sample(sample, transform):
    img = Image.open(io.BytesIO(sample["image"])).convert("RGB")
    if transform:
        img = transform(img)
    return img, int(sample["label"])

def make_collate(transform):
    def _collate(batch):
        xs, ys = zip(*(decode_sample(s, transform) for s in batch))
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)
    return _collate


train_ds = StreamingDataset(
    input_dir=TRAIN_URL,
    cache_dir=LITDATA_CACHE_DIR,
    shuffle=True,
    max_pre_download=MAX_PRE_DOWNLOAD,
)

val_ds = StreamingDataset(
    input_dir=VAL_URL,
    cache_dir=LITDATA_CACHE_DIR,
    shuffle=False,
    max_pre_download=MAX_PRE_DOWNLOAD,
)

test_ds = StreamingDataset(
    input_dir=TEST_URL,
    cache_dir=LITDATA_CACHE_DIR,
    shuffle=False,
    max_pre_download=MAX_PRE_DOWNLOAD,
)


train_loader = StreamingDataLoader(
    train_ds,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=DATALOADER_WORKERS,
    collate_fn=make_collate(train_transform),
    **({} if PREFETCH_FACTOR is None else {"prefetch_factor": PREFETCH_FACTOR}),
)

val_loader = StreamingDataLoader(
    val_ds,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=DATALOADER_WORKERS,
    collate_fn=make_collate(val_transform),
    **({} if PREFETCH_FACTOR is None else {"prefetch_factor": PREFETCH_FACTOR}),
)

test_loader = StreamingDataLoader(
    test_ds,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=DATALOADER_WORKERS,
    collate_fn=make_collate(val_transform),
    **({} if PREFETCH_FACTOR is None else {"prefetch_factor": PREFETCH_FACTOR}),
)


class Food11Model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")

        # freeze backbone
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
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, _):
        x, y = batch
        logits = self(x)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return optim.Adam(self.model.classifier.parameters(), lr=config["lr"])


model = Food11Model()

trainer = Trainer(
    accelerator="gpu",
    devices="auto",
    max_epochs=config["total_epochs"],
    num_sanity_val_steps=0,
    callbacks=[
        ModelCheckpoint(monitor="val_loss", mode="min"),
        EarlyStopping(monitor="val_loss", patience=config["patience"]),
    ],
)

trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)
