# --------------------------------------------------------
# WoundAmbit
# Copyright (c) 2025 Vanessa Borst, Timo Dittus and Contributors.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------


import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from beartype import beartype
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from medseg.evaluation.segmentation_visualizer import SegmentationVisualizer
from medseg.models.segmentors import SegformerB3, InternImageUperNet_T, TransNeXtUperNet_Tiny, VWFormerMiTB3, \
    VWFormerConvNextS, FCBFormer, HarDNetDFUS, SegNeXtL, FUSegNet, UNet, MISSFormer, HiFormerB
from medseg.util.path_builder import PathBuilder

IMG_PATH = PathBuilder().root().add("data").add("datasets").add("size_retrieval")
DATASET_NAME = "size_retrieval"
SAVE_CONTOURS = True
BATCH_SIZE = 1

# Important: This script expects the 5-fold CV models to be saved at `out/k_fold_models`.
# By default, separate inferences are run for each of the following models;
# If you want to use only some models, just comment the unwanted rows to deactivate these models
MODEL_MAPPING = {
    # ModelName: (Class, FolderName, Constructor Arguments based on the respective paper configs)
    "TransNeXt": (TransNeXtUperNet_Tiny, "transnextupernet_tiny", {"in_size": 512}),
    "InternImage": (InternImageUperNet_T, "internimageupernet_t", {"in_size": 512}),
    "VWFormerMiTB3": (VWFormerMiTB3, "vwformermitb3", {"in_size": 512}),
    "SegFormer": (SegformerB3, "segformerb3", {"in_size": 512}),
    "VWFormerConvNeXtS": (VWFormerConvNextS, "vwformerconvnexts", {"in_size": 512}),
    ###############
    "FCBFormer": (FCBFormer, "fcbformer", {"in_size": 512}),
    "HarDNet-DFUS": (HarDNetDFUS, "hardnetdfus", {"in_size": 512}),
    "SegNeXt": (SegNeXtL, "segnextl", {"in_size": 512}),
    "FuSegNet": (FUSegNet, "fusegnet", {"in_size": 512}),
    "UNet": (UNet, "unet", {"in_size": 512, "use_pretrained": True}),
    "MISSFormer": (MISSFormer, "missformer", {"in_size": 512, "encoder_pretrained": True, "operate_on_224": True}),
    "HiFormer": (HiFormerB, "hiformerb", {"in_size": 512}),
}

# Use the TOP 5 models based on the majority vote for the UKW OOD dataset
# Custom ensembles can be built by extending the following dictionary.
ENSEMBLES = {
    # EnsembleName : List of Models
    "Ensemble_TOP_5": ["TransNeXt", "InternImage", "VWFormerMiTB3", "SegFormer", "VWFormerConvNeXtS"],
}


class ImageDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".jpeg", ".png", ".JPG", ".png"))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        # Check if the image has square dimensions and if not, pad it to make it square
        original_size = image.size
        padding = (0, 0, 0, 0)
        if image.width != image.height:
            # Pad the image to make it square by adding zeros to the smaller side
            image, padding = self._pad_to_square(image)

        img_tensor = self.transform(image)
        return img_tensor, self.image_files[idx], original_size, padding

    def _pad_to_square(self, image):
        """Pads a non-square image to make it square by padding the smaller side with zeros."""
        width, height = image.size
        size = max(width, height)

        pad_w = (size - width) // 2
        pad_h = (size - height) // 2
        padding = (pad_w, pad_h, size - width - pad_w, size - height - pad_h)
        transform_pad = transforms.Pad(padding, fill=0)
        image = transform_pad(image)

        return image, padding


class ImageInferenceHelper:
    @beartype
    def __init__(self, model_names: list[str], model_ckpts_paths: dict[str, PathBuilder], img_path: PathBuilder,
                 dataset_name:str, is_kfold: bool = True, save_contours: bool = True,
                 batch_size: int = 8, num_workers: int = 4,
                 ensemble_name: str = "ensemble"):
        """
        Args:
            model_names (list[str]): List of model names for inference.
            model_ckpts_paths (dict): Mapping from model names to checkpoint paths.
            img_path (PathBuilder): Path to the dataset.
            dataset_name (str): Name of the dataset
            is_kfold (bool): Whether to use k-fold ensemble.
            save_contours (bool): Whether to overlay contours on output images.
            batch_size (int): Number of images processed at once.
            num_workers (int): Number of parallel data loading threads.
            ensemble_name (str): Name of the ensemble (if any).
        """
        self.model_names = model_names
        self.model_ckpts_paths = model_ckpts_paths

        # The number of models must match the number of checkpoint paths
        assert len(model_names) == len(model_ckpts_paths), "Number of models and checkpoint paths must match."

        self.img_path = img_path
        self.dataset_name = dataset_name
        self.is_kfold = is_kfold
        self.save_contours = save_contours
        self.ensemble_mode = len(model_names) > 1
        self.ensemble_name = ensemble_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=512, antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.dataset = ImageDataset(self.img_path.build(), self.transform)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,
                                     num_workers=self.num_workers, pin_memory=True)

        # Load models **once**
        self.models = self._load_all_models()

    def _load_all_models(self):
        """Loads all models into memory before inference."""
        models = {}
        for model_name in self.model_names:
            model_ckpts = self._get_model_checkpoints(model_name)
            models[model_name] = []

            for ckpt in model_ckpts:
                model = self._load_model(model_name, ckpt)
                models[model_name].append((os.path.basename(os.path.dirname(ckpt)),model))

        return models

    def _load_model(self, model_name: str, checkpoint_path: str):
        """Loads a model from a checkpoint and moves it to GPU."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Retrieve the model class, identifier, and constructor arguments from the dictionary
        model_class, _, init_args = MODEL_MAPPING[model_name]
        # Create a new instance of the model by unpacking the arguments
        model = model_class(**init_args)  # Pass arguments dynamically

        model.load_state_dict(checkpoint["model"])
        model.to(self.device)
        model.eval()

        return model

    def _get_model_checkpoints(self, model_name: str):
        """Returns a list of checkpoint paths for a given model (k-fold inference)."""
        model_ckpt_path = self.model_ckpts_paths[model_name]
        if self.is_kfold:
            return [os.path.join(model_ckpt_path.build(), fold, "best_checkpoint.pt")
                    for fold in os.listdir(model_ckpt_path.build())
                    if os.path.isdir(os.path.join(model_ckpt_path.build(), fold))]
        else:
            return [model_ckpt_path.build()]

    def run_inference(self):
        """Runs inference on images in batches using multiple models."""
        for _, (img_tensors, image_names, original_sizes, paddings) in enumerate(self.dataloader):
            img_tensors = img_tensors.to(self.device)

            all_predictions = []
            for model_name in self.models:
                for fold_name, model in self.models[model_name]:  # Iterate through k-fold models

                    with torch.no_grad():
                        pred = model(img_tensors)  # (Batch, C, H, W)
                        out = torch.sigmoid(pred)  # Model returns logits

                    binary_out = (out > 0.5).float()  # Apply threshold
                    all_predictions.append(binary_out)

            # Perform majority voting for the batch
            final_masks = self._majority_voting(all_predictions)  # Shape: (batch_size, H, W)

            # Reshape paddings and original sizes to shape batch_size, tuple
            # where the tuple contains either 4 elements (left, top, right, bottom) or two elements (width, height)
            paddings_tuples = [tuple(int(x.item()) for x in padding) for padding in zip(*paddings)]
            original_sizes_tuples = [tuple(int(x.item()) for x in size) for size in zip(*original_sizes)]

            for i, image_name in enumerate(image_names):
                # Undo padding and rescale
                final_mask = self._undo_padding_and_rescaling(final_masks[i], paddings_tuples[i],
                                                              original_sizes_tuples[i])

                # Convert to uint8 (0/255) for saving as an image
                final_mask = (final_mask * 255).to(torch.uint8).squeeze().cpu().numpy()
                mask_img = Image.fromarray(final_mask)  # Convert mask to grayscale image

                # Overlay contours on the mask
                if self.save_contours:
                    image_path = os.path.join(self.img_path.build(), image_name)
                    annotated_image = SegmentationVisualizer.draw_mask_contour(Image.open(image_path), mask_img,
                                                                               color=(0, 255, 0))
                else:
                    annotated_image = None

                # Save the mask
                self._save_mask(image_name, mask_img, annotated_image)

    @staticmethod
    def _majority_voting(predictions: list):
        """
        Performs pixel-wise majority voting for each image in a batch.

        Args:
            predictions (list): List of model predictions.
                                Shape of each entry: (batch_size, 1, H, W)
                                Number of entries = num_models.

        Returns:
            np.ndarray: Majority-voted segmentation masks for the batch.
                        Shape: (batch_size, H, W)
        """
        # Stack predictions along a new axis (num_models, batch_size, 1, H, W)
        stacked_preds = torch.cat(predictions, dim=1)  # Shape: (batch_size, num_models, 1, H, W)

        # Majority voting along axis=1 (num_models dimension)
        # Compute majority vote using bincount
        majority_mask = torch.mode(stacked_preds, dim=1).values  # Shape: (batch_size, H, W)

        # Slow:
        # Convert to NumPy for stats.mode (requires (num_models, batch_size, H, W))
        # mask_preds_int = stacked_preds.squeeze(2).cpu().numpy().astype(int)  # Shape: (batch_size, num_models, H, W)
        # majority_mask, _ = stats.mode(mask_preds_int, axis=1, keepdims=False)  # Shape: (batch_size, H, W)

        return majority_mask  # Final shape: (batch_size, H, W)

    @staticmethod
    def _undo_padding_and_rescaling(mask, padding, original_size):
        """
        Crops the mask back to the original image size after padding.

        Args:
            mask (np.ndarray): The predicted segmentation mask.
            padding (tuple): Padding applied (left, top, right, bottom).
            original_size (tuple): Original (width, height) before padding.

        Returns:
            np.ndarray: Cropped and resized mask.
        """
        orig_width, orig_height = original_size
        padded_size = max(orig_width, orig_height)

        # Add batch dim, mask is already a float tensor
        mask = mask.unsqueeze(0).unsqueeze(0)
        # Resize to the original dimensions AFTER padding
        mask = F.interpolate(mask, size=(padded_size, padded_size), mode="bilinear", align_corners=False)

        # Crop out the padded region
        if padding is not None:
            left, top, right, bottom = padding
            mask = mask[:, :, top:padded_size - bottom, left:padded_size - right]

        # Thresholding to binary mask
        mask = (mask > 0.5).float()

        return mask

    def _save_mask(self, image_name, mask_img, annotated_image=None):
        """Saves the predicted mask and also the annotated image if one is passed."""
        model_folder = self.ensemble_name if self.ensemble_mode else self.model_names[0]
        save_path = PathBuilder().out().add("predictions").add(self.dataset_name).add(model_folder)

        save_path = os.path.join(save_path.build(),f"{image_name.split('.')[0]}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mask_img.save(save_path)

        print(f"Saved mask for {image_name} at {save_path}")

        if annotated_image is not None:
            save_path = PathBuilder().out().add("predictions").add(self.dataset_name).add(f"{model_folder}_annotated")
            save_path = os.path.join(save_path.build(), f"{image_name.split('.')[0]}_annotated.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            annotated_image.save(save_path)
            print(f"Saved mask contours for {image_name} at {save_path}")


if __name__ == "__main__":

    # Single-model inference
    for model_name in MODEL_MAPPING.keys():
        print(f"Running inference for model: {model_name} with batch size {BATCH_SIZE}")
        inference = ImageInferenceHelper(
            model_names=[model_name],
            model_ckpts_paths={
                model_name: PathBuilder().root().out().add("k_fold_models").add(MODEL_MAPPING[model_name][1])},
            img_path=IMG_PATH,
            dataset_name=DATASET_NAME,
            is_kfold=True,
            save_contours=SAVE_CONTOURS,
            batch_size=BATCH_SIZE,
            num_workers=4
        )
        inference.run_inference()

    # Repeat the inference for different sets of ensembles
    for ensemble_name, model_names in ENSEMBLES.items():
        BATCH_SIZE = 20 if ensemble_name == "Ensemble_All" else 50
        print(f"Running ensemble inference for {ensemble_name} with batch size {BATCH_SIZE}"
              f" and {len(model_names)} models.")
        inference_ensemble = ImageInferenceHelper(
            model_names=model_names,
            model_ckpts_paths={name: PathBuilder().root().out().add("k_fold_models").add(MODEL_MAPPING[name][1]) for
                               name in model_names},
            img_path=IMG_PATH,
            dataset_name=DATASET_NAME,
            is_kfold=True,
            save_contours=SAVE_CONTOURS,
            batch_size=BATCH_SIZE,
            num_workers=4,
            ensemble_name=ensemble_name
        )
        inference_ensemble.run_inference()
