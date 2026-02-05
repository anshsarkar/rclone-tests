import os
import io
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L

from PIL import Image
from torchvision import models, transforms
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
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
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
CACHE_DIR = os.getenv("LITDATA_CACHE_DIR", "/tmp/litdata_cache")

TRAIN_URL = f"s3://{S3_BUCKET}/{LITDATA_PREFIX}/training"
VAL_URL   = f"s3://{S3_BUCKET}/{LITDATA_PREFIX}/validation"
TEST_URL  = f"s3://{S3_BUCKET}/{LITDATA_PREFIX}/evaluation"

STORAGE_OPTIONS = {"endpoint_url": os.environ["S3_ENDPOINT_URL"]}
SESSION_OPTIONS = {
    "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
    "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
    "region_name": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
}

NUM_WORKERS = int(os.getenv("DATALOADER_WORKERS", "0"))
PREFETCH = int(os.getenv("PREFETCH_FACTOR", "2")) if NUM_WORKERS > 0 else None
MAX_PRE_DOWNLOAD = int(os.getenv("LITDATA_MAX_PRE_DOWNLOAD", "16"))

def decode(sample, transform):
    img = Image.open(io.BytesIO(sample["image"])).convert("RGB")
    return transform(img), int(sample["label"])

def collate_fn(transform):
    def _fn(batch):
        xs, ys = zip(*(decode(s, transform) for s in batch))
        return torch.stack(xs), torch.tensor(ys)
    return _fn

train_ds = StreamingDataset(
    input_dir=TRAIN_URL,
    cache_dir=CACHE_DIR,
    shuffle=True,
    max_pre_download=MAX_PRE_DOWNLOAD,
    storage_options=STORAGE_OPTIONS,
    session_options=SESSION_OPTIONS,
)

val_ds = StreamingDataset(
    input_dir=VAL_URL,
    cache_dir=CACHE_DIR,
    shuffle=False,
    max_pre_download=MAX_PRE_DOWNLOAD,
    storage_options=STORAGE_OPTIONS,
    session_options=SESSION_OPTIONS,
)

test_ds = StreamingDataset(
    input_dir=TEST_URL,
    cache_dir=CACHE_DIR,
    shuffle=False,
    max_pre_download=MAX_PRE_DOWNLOAD,
    storage_options=STORAGE_OPTIONS,
    session_options=SESSION_OPTIONS,
)

train_loader = StreamingDataLoader(
    train_ds,
    batch_size=config["batch_size"],
    num_workers=NUM_WORKERS,
    shuffle=True,
    collate_fn=collate_fn(train_transform),
    **({} if PREFETCH is None else {"prefetch_factor": PREFETCH}),
)

val_loader = StreamingDataLoader(
    val_ds,
    batch_size=config["batch_size"],
    num_workers=NUM_WORKERS,
    shuffle=False,
    collate_fn=collate_fn(val_transform),
    **({} if PREFETCH is None else {"prefetch_factor": PREFETCH}),
)

test_loader = StreamingDataLoader(
    test_ds,
    batch_size=config["batch_size"],
    num_workers=NUM_WORKERS,
    shuffle=False,
    collate_fn=collate_fn(val_transform),
    **({} if PREFETCH is None else {"prefetch_factor": PREFETCH}),
)

class LightningFood11Model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")

        for p in self.model.features.parameters():
            p.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Dropout(config["dropout_probability"]),
            nn.Linear(self.model.last_channel, 11),
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        acc = (out.argmax(1) == y).float().mean()
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        acc = (out.argmax(1) == y).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.model.classifier.parameters(), lr=config["lr"])

model = LightningFood11Model()

trainer = Trainer(
    max_epochs=config["total_epochs"],
    accelerator="gpu",
    devices="auto",
    num_sanity_val_steps=0,
    callbacks=[
        ModelCheckpoint(monitor="val_loss", mode="min"),
        EarlyStopping(monitor="val_loss", patience=config["patience"]),
    ],
)

trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)
