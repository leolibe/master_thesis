from pathlib import Path
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array


def load_image_and_mask(image_path, mask_path, input_size=(512, 512)):
    image = load_img(image_path, target_size=input_size, color_mode="rgb")
    image = img_to_array(image) / 255.0

    mask = load_img(mask_path, target_size=input_size, color_mode="grayscale")
    mask = img_to_array(mask)
    mask = (mask > 0).astype(np.uint8)

    return image.astype(np.float32), mask


def make_overlay(image_uint8, mask_2d, alpha=0.35, color=(255, 255, 255)):
    mask_2d = (mask_2d > 0).astype(np.uint8)

    color_mask = np.zeros_like(image_uint8)
    color_mask[:, :] = color

    overlay = np.where(
        mask_2d[:, :, None] > 0,
        (1 - alpha) * image_uint8 + alpha * color_mask,
        image_uint8
    )

    return overlay.astype(np.uint8)


def plot_image_mask_pair(image_path, mask_path, input_size=(512, 512)):
    image, mask = load_image_and_mask(image_path, mask_path, input_size)

    image_uint8 = (image * 255).astype(np.uint8)
    mask_2d = mask.squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image_uint8)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask_2d, cmap="gray")
    axes[1].set_title("Ground truth")
    axes[1].axis("off")

    axes[2].imshow(make_overlay(image_uint8, mask_2d))
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def plot_predictions_from_files(
    model,
    image_dir,
    mask_dir,
    image_filenames,
    mask_filenames,
    input_size=(512, 512),
    n_samples=10,
    seed=42,
    threshold=0.5,
    save_path=None
):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    if len(image_filenames) != len(mask_filenames):
        raise ValueError("image_filenames and mask_filenames must have the same length")

    n_samples = min(n_samples, len(image_filenames))

    random.seed(seed)
    indices = random.sample(range(len(image_filenames)), n_samples)

    selected_images = [image_filenames[i] for i in indices]
    selected_masks = [mask_filenames[i] for i in indices]

    images = []
    gt_masks = []

    for img_name, mask_name in zip(selected_images, selected_masks):
        image, mask = load_image_and_mask(
            image_dir / img_name,
            mask_dir / mask_name,
            input_size=input_size
        )
        images.append(image)
        gt_masks.append(mask)

    images = np.array(images, dtype=np.float32)
    gt_masks = np.array(gt_masks, dtype=np.uint8)

    preds = model.predict(images, verbose=1)
    preds = (preds > threshold).astype(np.uint8)

    fig, axes = plt.subplots(n_samples, 4, figsize=(14, 3 * n_samples))

    if n_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n_samples):
        image_uint8 = (images[i] * 255).astype(np.uint8)

        gt_mask = gt_masks[i].squeeze()
        pred_mask = preds[i].squeeze()

        pred_overlay = make_overlay(image_uint8, pred_mask, alpha=0.35)
        gt_overlay = make_overlay(image_uint8, gt_mask, alpha=0.35)

        axes[i, 0].imshow(image_uint8)
        axes[i, 0].set_title(f"Image\n{selected_images[i]}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt_mask, cmap="gray")
        axes[i, 1].set_title("Ground truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_mask, cmap="gray")
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(pred_overlay)
        axes[i, 3].set_title("Predicted overlay")
        axes[i, 3].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.show()


def plot_training_history(history, save_dir=None):
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.history["loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history["loss"], label="Train loss")
    plt.plot(epochs, history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir / "training_loss.png", bbox_inches="tight", dpi=150)

    plt.show()

    if "accuracy" in history.history and "val_accuracy" in history.history:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history.history["accuracy"], label="Train accuracy")
        plt.plot(epochs, history.history["val_accuracy"], label="Validation accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and validation accuracy")
        plt.legend()
        plt.tight_layout()

        if save_dir is not None:
            plt.savefig(save_dir / "training_accuracy.png", bbox_inches="tight", dpi=150)

        plt.show()

    if "dice_coef" in history.history and "val_dice_coef" in history.history:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history.history["dice_coef"], label="Train Dice")
        plt.plot(epochs, history.history["val_dice_coef"], label="Validation Dice")
        plt.xlabel("Epoch")
        plt.ylabel("Dice coefficient")
        plt.title("Training and validation Dice")
        plt.legend()
        plt.tight_layout()

        if save_dir is not None:
            plt.savefig(save_dir / "training_dice.png", bbox_inches="tight", dpi=150)

        plt.show()