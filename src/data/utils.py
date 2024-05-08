import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union

import torch
import numpy as np
from PIL import Image
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download

DATASET_ID = "EduardoPacheco/FoodSeg103"
NUM_CLASSES = 104 # 103 classes + 1 background class

@dataclass
class DataTrainingArguments:
    mask_ratio: float = field(default=0.75)
    random_coloring: bool = field(default=True)
    num_pairs_per_image: Optional[int] = field(default=None)
    # train_batch_size: int = field(default=8)
    # validation_batch_size: int = field(default=8)
    # num_workers: int = field(default=4)
    # pin_memory: bool = field(default=True)
    # num_epochs: int = field(default=1)

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

    return mask.bool()

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for the FoodSeg103 dataset."""
    keys = batch[0].keys()
    collate_batch = {}
    for key in keys:
        collate_batch[key] = torch.stack([sample[key] for sample in batch])
    
    return collate_batch

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