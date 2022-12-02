import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_type = "clothes"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/" + train_type + "/train"
VAL_DIR = "data/" + train_type + "/train"
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 50
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = train_type + ".disc.pth.tar"
CHECKPOINT_GEN = train_type + ".gen.pth.tar"

both_transform = A.Compose(
    [A.Resize(width=256, height=256), ], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        # A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
        ToTensorV2(),
    ]
)
