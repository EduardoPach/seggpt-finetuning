import json
import math
import itertools
from typing import List, Tuple, Dict, Any, Optional

import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download
from transformers import SegGptImageProcessor, SegGptConfig
from torchvision.transforms import Compose, ToTensor, Normalize

DATASET_ID = "EduardoPacheco/FoodSeg103"
NUM_CLASSES = 104 # 103 classes + 1 background class


def load_foodseg103(split: str) -> Dataset:
    """Loads the FoodSeg103 dataset using the Hugging Face datasets library."""
    return load_dataset(DATASET_ID, split=split)

def get_foodseg103_id2label() -> Dict[int, str]:
    """Returns the labels for the FoodSeg103 dataset."""
    id2label = json.load(open(hf_hub_download(DATASET_ID, "id2label.json", repo_type="dataset"), "r"))
    return {int(k): v for k, v in id2label.items()}

def random_color_palette(indicies: List[int]) -> Dict[int, np.array]:
    """Generates a random color palette for coloring masks."""
    palette = {}
    for idx in indicies:
        palette[idx] = np.random.randint(0, 256, 3)
    return palette

def mask_coloring(mask: Image.Image, palette: Optional[Dict[int, np.array]]=None) -> Image.Image:
    """Colorizes the mask using a palette."""
    mask_array = np.array(mask)
    classes = np.unique(mask_array)
    colored_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
    for cls in classes:
        # Skip the background class as it should be black
        if cls == 0:
            continue
        color = palette[cls]
        colored_mask[mask_array == cls] = color
    
    return Image.fromarray(colored_mask)

def random_masking(num_patches: int, mask_ratio: float = 0.75, is_train:Optional[bool] = None) -> torch.BoolTensor:
    """Generates a random booled mask with a given ratio of masked patches with shape (num_patches,)."""
    if not is_train:
        return None
    
    num_masked_patches = int(num_patches * mask_ratio)
    shuffle_idx = torch.randperm(num_patches)
    mask = torch.FloatTensor([0] * (num_patches - num_masked_patches) + [1] * num_masked_patches)[shuffle_idx]

    return mask

def load_foodseg103_for_incontext_tuning(
    config: SegGptConfig, 
    image_processor: SegGptImageProcessor,
    split: str = "train",
    mask_ratio: float = 0.75,
    random_coloring: bool = True
) -> Dataset:
    """Loads the FoodSeg103 dataset for in-context tuning using the Hugging Face datasets library."""
    ds = load_dataset(DATASET_ID, split=split)
    num_patches = math.prod(i // config.patch_size for i in config.image_size)
    is_train = split == "train"

    if random_coloring:
        to_tensor = ToTensor()

        return ds.map(lambda example: {
            "labels": to_tensor(mask_coloring(example["label"])),
            "pixel_values": image_processor(images=example["image"])["pixel_values"],
            "bool_masked_pos": random_masking(num_patches, mask_ratio=mask_ratio, is_train=is_train)
            }
        )
    
    return ds.map(lambda example: {
        "labels": image_processor(images=None, prompt_masks=example["label"], num_labels=NUM_CLASSES - 1)["prompt_masks"],
        "pixel_values": image_processor(images=example["image"])["pixel_values"],
        "bool_masked_pos": random_masking(num_patches, mask_ratio=mask_ratio, is_train=is_train)
        }
    )
    




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

#TODO finish this
class FoodSegDataset(Dataset):
    """
    Dataset for FoodSeg103 for fine-tuning SegGPT using the Hugging Face datasets library.
    """
    def __init__(
        self, 
        split: str,
        config: SegGptConfig,
        image_processor: SegGptImageProcessor, 
        mask_ratio: float = 0.75,
        transform=None
    ) -> None:
        self.split = split
        self.config = config
        self.transform = transform
        self.dataset_id = DATASET_ID
        self.mask_ratio = mask_ratio
        self.is_train = split == 'train'
        self.image_processor = image_processor

        self.ds = load_dataset(self.dataset_id, split=self.split)
        self.pairs = self.set_image_pairs()

    def set_image_pairs(self) -> List[Tuple[int, int]]:
        classes = [set(cls) for cls in self.ds["classes_on_image"]]
        indicies = list(range(len(classes)))

        # When training combine and then add randomness to swap the order
        # When evaluating get all possible pairs for robust evaluation
        if self.is_train:
            pairs = itertools.combinations(indicies, 2)
        else: 
            pairs = itertools.permutations(indicies, 2)

        # To be a valid pair should have at least one class in common
        valid_pairs = []
        for i, j in pairs:
            intersection = classes[i].intersection(classes[j])
            if len(intersection) > 0:
                valid_pairs.append((i, j))
        
        return valid_pairs
    
    def random_masking(self) -> torch.BoolTensor:
        ...

    def random_coloring(self, mask) -> Tuple[Image.Image, Image.Image]:
        ...

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample