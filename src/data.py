import json
import math
import random
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Union

import torch
import numpy as np
from PIL import Image
from datasets import load_dataset, Dataset
from torch.utils.data import Dataset as PyTorchDataset
from huggingface_hub import hf_hub_download
from transformers import SegGptImageProcessor, SegGptConfig

DATASET_ID = "EduardoPacheco/FoodSeg103"
NUM_CLASSES = 104 # 103 classes + 1 background class

@dataclass
class DataTrainingArguments:
    mask_ratio: float = field(default=0.75)
    random_coloring: bool = field(default=True)

def load_foodseg103(split: str) -> Dataset:
    """Loads the FoodSeg103 dataset using the Hugging Face datasets library."""
    return load_dataset(DATASET_ID, split=split)

def get_foodseg103_id2label() -> Dict[int, str]:
    """Returns the labels for the FoodSeg103 dataset."""
    id2label = json.load(open(hf_hub_download(DATASET_ID, "id2label.json", repo_type="dataset"), "r"))
    return {int(k): v for k, v in id2label.items()}

def random_color_palette(indicies: Union[List[int], np.array]) -> Dict[int, np.array]:
    """Generates a random color palette for coloring masks."""
    palette = {}
    used_colors = set()  # Keep track of used colors

    for idx in indicies:
        color = np.random.randint(0, 256, 3)  # Generate a random color
        while tuple(color) == (0, 0, 0) or tuple(color) in used_colors:  # Check if color is (0, 0, 0) or already used
            color = np.random.randint(0, 256, 3)  # Generate a new color
        palette[idx] = color
        used_colors.add(tuple(color))  # Add color to used colors set

    return palette

def mask_coloring(mask: Image.Image, palette: Optional[Dict[int, np.array]]=None) -> Image.Image:
    """Colorizes the mask using a palette."""
    mask_array = np.array(mask)
    classes = np.unique(mask_array)
    if palette is None:
        palette = random_color_palette(classes)
    colored_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
    for cls in classes:
        # Skip the background class as it should be black
        if cls == 0:
            continue
        color = palette[cls]
        colored_mask[mask_array == cls] = color
    
    return Image.fromarray(colored_mask)

def random_masking(num_patches: int, mask_ratio: float = 0.75) -> torch.BoolTensor:
    """Generates a random booled mask with a given ratio of masked patches with shape (num_patches,)."""
    num_masked_patches = int(num_patches * mask_ratio)
    shuffle_idx = torch.randperm(num_patches)
    mask = torch.FloatTensor([0] * (num_patches - num_masked_patches) + [1] * num_masked_patches)[shuffle_idx]

    return mask

class TransformFineTuning:
    def __init__(
        self,
        ds: Dataset,
        config: SegGptConfig,
        image_processor: SegGptImageProcessor,
        mask_ratio: float = 0.75,
        random_coloring: bool = True,
        is_train: bool = True,
        pair_mapping: Optional[Dict[int, List[int]]] = None
    ) -> None:
        self.ds = ds
        self.config = config
        self.image_processor = image_processor
        self.mask_ratio = mask_ratio
        self.random_coloring = random_coloring
        self.is_train = is_train
        self.pair_mapping = pair_mapping if pair_mapping else self._set_pairs()

    def _set_pairs(self) -> None:
        classes = [set(cls) for cls in self.ds["classes_on_image"]]
        ids = self.ds["id"]

        pairs = itertools.combinations(ids, 2)

        # To be a valid pair should have at least one class in common
        pairs_mapping = defaultdict(list)
        for i, j in pairs:
            intersection = classes[i].intersection(classes[j])
            if len(intersection) > 0:
                pairs_mapping[i].append(j)
                pairs_mapping[j].append(i)
        
        return pairs_mapping


    def __call__(self, example: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        num_patches = math.prod(i // self.config.patch_size for i in self.config.image_size)

        sample_ids = example["id"]
        pair_ids = [random.choice(self.pair_mapping[sample_id]) for sample_id in sample_ids]

        labels = example["label"]
        images = example["image"]

        batch_size = len(images)
        bool_masked_pos = None
        if self.is_train:
            bool_masked_pos = [random_masking(num_patches, mask_ratio=self.mask_ratio) for _ in range(batch_size)]
            bool_masked_pos = torch.stack(bool_masked_pos)

        prompt_images = self.ds[pair_ids]["image"]
        prompt_masks = self.ds[pair_ids]["label"]

        if self.random_coloring:
            palette = random_color_palette(list(range(1, NUM_CLASSES)))
            prompt_masks = [mask_coloring(prompt_mask, palette) for prompt_mask in prompt_masks]
            labels = [mask_coloring(label, palette) for label in labels]

        inputs = self.image_processor(
            images=images, 
            prompt_masks=prompt_masks,
            prompt_images=prompt_images,
            num_labels=NUM_CLASSES - 1,
            do_convert_rgb=not self.random_coloring, 
            return_tensors="pt"
        )

        labels_inputs = self.image_processor(
            images=None,
            prompt_masks=labels,
            num_labels=NUM_CLASSES - 1,
            do_convert_rgb=not self.random_coloring,
            return_tensors="pt"
        )

        return {
            "pixel_values": inputs["pixel_values"],
            "prompt_pixel_values": inputs["prompt_pixel_values"],
            "prompt_masks": inputs["prompt_masks"],
            "labels": labels_inputs["prompt_masks"],
            "bool_masked_pos": bool_masked_pos
        }

class TransformInContextTuning:
    def __init__(
        self,
        config: SegGptConfig,
        image_processor: SegGptImageProcessor,
        mask_ratio: float = 0.75,
        random_coloring: bool = True,
        is_train: bool = True
    ) -> None:
        self.config = config
        self.image_processor = image_processor
        self.mask_ratio = mask_ratio
        self.random_coloring = random_coloring
        self.is_train = is_train
    
    def __call__(self, example: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        num_patches = math.prod(i // self.config.patch_size for i in self.config.image_size)

        labels = example["label"]
        images = example["image"]

        batch_size = len(images)

        bool_masked_pos = None
        if self.is_train:
            bool_masked_pos = [random_masking(num_patches, mask_ratio=self.mask_ratio) for _ in range(batch_size)]
            bool_masked_pos = torch.stack(bool_masked_pos)

        inputs = self.image_processor(
            images=images, 
            prompt_masks=mask_coloring(labels) if self.random_coloring else labels, 
            num_labels=NUM_CLASSES - 1, 
            do_convert_rgb=not self.random_coloring,
            return_tensors="pt"
        )

        return {
            "pixel_values": inputs["pixel_values"],
            "labels": inputs["prompt_masks"],
            "bool_masked_pos": bool_masked_pos
        }
    

def get_in_context_datasets(config, image_processor, mask_ratio=0.75, random_coloring=True) -> Tuple[Dataset, Dataset]:
    """Returns the training and validation datasets for the FoodSeg103 dataset with their respective transformations."""
    ds_train = load_foodseg103("train")
    ds_val = load_foodseg103("validation")

    transform_train = TransformInContextTuning(config, image_processor, mask_ratio, random_coloring, is_train=True)
    transform_val = TransformInContextTuning(config, image_processor, mask_ratio, random_coloring, is_train=False)

    ds_train.set_transform(transform_train)
    ds_val.set_transform(transform_val)

    return ds_train, ds_val

def get_fine_tuning_datasets(config, image_processor, mask_ratio=0.75, random_coloring=True) -> Tuple[Dataset, Dataset]:
    """Returns the training and validation datasets for the FoodSeg103 dataset with their respective transformations."""
    ds_train = load_foodseg103("train")
    train_dummy = load_foodseg103("train") # Need the dummy otherwise will recurse infinitely

    ds_val = load_foodseg103("validation")
    val_dummy = load_foodseg103("validation") # Need the dummy otherwise will recurse infinitely

    transform_train = TransformFineTuning(train_dummy, config, image_processor, mask_ratio, random_coloring, is_train=True)
    transform_val = TransformFineTuning(val_dummy, config, image_processor, mask_ratio, random_coloring, is_train=False)

    ds_train.set_transform(transform_train)
    ds_val.set_transform(transform_val)

    return ds_train, ds_val
    
class FoodSegLabelMapper:
    """Converts between label ids and labels for the FoodSeg103 dataset."""
    def __init__(self) -> None:
        self.id2label = get_foodseg103_id2label()
        self.label2id = {v: k for k, v in self.id2label.items()}

    def __len__(self) -> int:
        return len(self.id2label)
    
    @property
    def labels(self) -> List[str]:
        return list(self.id2label.values())
    
    def get_label(self, idx: int) -> str:
        return self.id2label[idx]
    
    def get_id(self, label: str) -> int:
        return self.label2id[label]
    
if __name__ == "__main__":
    ds_train = load_foodseg103("train")
    train_dummy = load_foodseg103("train") # Need the dummy otherwise will recurse infinitely

    transforms = TransformFineTuning(
        train_dummy, 
        SegGptConfig.from_pretrained("BAAI/seggpt-vit-large"), 
        SegGptImageProcessor.from_pretrained("BAAI/seggpt-vit-large"),
        mask_ratio=0.75,
        random_coloring=True,
        is_train=True
    )

    ds_train.set_transform(transforms)

