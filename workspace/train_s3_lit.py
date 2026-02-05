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
    "epochs": 20,
    "batch_size": int(os.getenv("BATCH_SIZE", "32")),
    "lr": 1e-4,
    "dropout": 0.5,
}

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

S3_BUCKET = os.getenv("S3_BUCKET", "rb-litdata-food11")
PREFIX = os.getenv("LITDATA_PREFIX", "litdata_food11")
CACHE_DIR = os.getenv("LITDATA_CACHE_DIR", "/tmp/litdata_cache")

TRAIN_URL = f"s3://{S3_BUCKET}/{PREFIX}/training"
VAL_URL   = f"s3://{S3_BUCKET}/{PREFIX}/validation"
TEST_URL  = f"s3://{S3_BUCKET}/{PREFIX}/evaluation"

STORAGE_OPTIONS = {"endpoint_url": os.environ["S3_ENDPOINT_URL"]}
SESSION_OPTIONS = {
    "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
    "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
    "region_name": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
}

NUM_WORKERS = int(os.getenv("DATALOADER_WORKERS", "8"))
PREFETCH = int(os.getenv("PREFETCH_FACTOR", "4"))
MAX_PRE_DOWNLOAD = int(os.getenv("LITDATA_MAX_PRE_DOWNLOAD", "4096"))

def decode(sample, transform):
    img = Image.open(io.BytesIO(sample["image"])).convert("RGB")
    return transform(img), int(sample["label"])

def collate_fn(transform):
    def _fn(batch):
        xs, ys = zip(*(decode(s, transform) for s in batch))
        return torch.stack(xs), torch.tensor(ys)
    return _fn

train_ds = StreamingDataset(
    TRAIN_URL, CACHE_DIR, shuffle=True,
    max_pre_download=MAX_PRE_DOWNLOAD,
    storage_options=STORAGE_OPTIONS,
    session_options=SESSION_OPTIONS,
    use_index=False
)

val_ds = StreamingDataset(
    VAL_URL, CACHE_DIR, shuffle=False,
    max_pre_download=MAX_PRE_DOWNLOAD,
    storage_options=STORAGE_OPTIONS,
    session_options=SESSION_OPTIONS,
    use_index=False
)

test_ds = StreamingDataset(
    TEST_URL, CACHE_DIR, shuffle=False,
    max_pre_download=MAX_PRE_DOWNLOAD,
    storage_options=STORAGE_OPTIONS,
    session_options=SESSION_OPTIONS,
    use_index=False
)

train_loader = StreamingDataLoader(
    train_ds,
    batch_size=config["batch_size"],
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH,
    shuffle=True,
    collate_fn=collate_fn(train_transform),
    
)

val_loader = StreamingDataLoader(
    val_ds,
    batch_size=config["batch_size"],
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH,
    shuffle=False,
    collate_fn=collate_fn(eval_transform),
    
)

test_loader = StreamingDataLoader(
    test_ds,
    batch_size=config["batch_size"],
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH,
    shuffle=False,
    collate_fn=collate_fn(eval_transform),
    
)

class Food11Model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")

        for p in self.model.features.parameters():
            p.requires_grad = False

        self.model.classifier = nn.Sequential(
            nn.Dropout(config["dropout"]),
            nn.Linear(self.model.last_channel, 11),
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        return loss, acc

    def training_step(self, batch, _):
        loss, acc = self.step(batch)
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, acc = self.step(batch)
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    def test_step(self, batch, _):
        loss, acc = self.step(batch)
        self.log_dict({"test_loss": loss, "test_acc": acc})

    def configure_optimizers(self):
        return optim.Adam(self.model.classifier.parameters(), lr=config["lr"])

model = Food11Model()

trainer = Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=config["epochs"],
    callbacks=[
        ModelCheckpoint(monitor="val_loss", mode="min"),
        EarlyStopping(monitor="val_loss", patience=3),
    ],
    num_sanity_val_steps=0,
)

trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)