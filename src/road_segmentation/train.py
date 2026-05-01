from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers.legacy import Adam
from road_segmentation.losses import bce_dice_loss, dice_coef, iou_score
from road_segmentation.dataset import make_dataset
from unet import unet
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2DTranspose
import yaml
from pathlib import Path

class Conv2DTransposeFixed(Conv2DTranspose):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


def load_config(path="configs/train_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_fixed(model_path):
    return load_model(
        model_path,
        custom_objects={"Conv2DTranspose": Conv2DTransposeFixed},
        compile=False
    )

def train_model(train_ds, val_ds, input_shape):
    model = unet(input_shape, output_layer=1)
    cfg = load_config()
    model.compile(
        optimizer=Adam(learning_rate=cfg["model"]["learning_rate"]),
        loss=bce_dice_loss,
        metrics=[dice_coef, iou_score]
    )

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(patience=3),
        ModelCheckpoint("best_model.h5", save_best_only=True)
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=callbacks
    )

    return model