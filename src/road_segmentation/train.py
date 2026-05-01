from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers.legacy import Adam
from losses import bce_dice_loss, dice_coef, iou_score
from dataset import make_dataset
from unet import unet
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2DTranspose


class Conv2DTransposeFixed(Conv2DTranspose):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


def load_model_fixed(model_path):
    return load_model(
        model_path,
        custom_objects={"Conv2DTranspose": Conv2DTransposeFixed},
        compile=False
    )

def train_model(train_ds, val_ds, input_shape):
    model = unet(input_shape, output_layer=1)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
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
        epochs=50,
        callbacks=callbacks
    )

    return model