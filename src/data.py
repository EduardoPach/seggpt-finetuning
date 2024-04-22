import json
import math
import itertools
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
    for idx in indicies:
        palette[idx] = np.random.randint(0, 256, 3)
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

def random_masking(num_patches: int, mask_ratio: float = 0.75, is_train:Optional[bool] = None) -> torch.BoolTensor:
    """Generates a random booled mask with a given ratio of masked patches with shape (num_patches,)."""
    if not is_train:
        return None
    
    num_masked_patches = int(num_patches * mask_ratio)
    shuffle_idx = torch.randperm(num_patches)
    mask = torch.FloatTensor([0] * (num_patches - num_masked_patches) + [1] * num_masked_patches)[shuffle_idx]

    return mask.unsqueeze(0)

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
    
    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        num_patches = math.prod(i // self.config.patch_size for i in self.config.image_size)
        bool_masked_pos = random_masking(num_patches, mask_ratio=self.mask_ratio, is_train=self.is_train)

        labels = example["label"][0]
        image = example["image"][0]

        # SegGptImageProcessor won't try to conver the mask to rgb if the mask is already in rgb
        inputs = self.image_processor(
            images=image, 
            prompt_masks=mask_coloring(labels) if self.random_coloring else labels, 
            num_labels=NUM_CLASSES - 1, 
            return_tensors="pt"
        )

        return {
            "pixel_values": inputs["pixel_values"],
            "labels": inputs["prompt_masks"],
            "bool_masked_pos": bool_masked_pos
        }
    

def get_datasets(config, image_processor, mask_ratio=0.75, random_coloring=True) -> Tuple[Dataset, Dataset]:
    """Returns the training and validation datasets for the FoodSeg103 dataset with their respective transformations."""
    transform_train = TransformInContextTuning(config, image_processor, mask_ratio, random_coloring, is_train=True)
    transform_val = TransformInContextTuning(config, image_processor, mask_ratio, random_coloring, is_train=False)

    ds_train = load_foodseg103("train")
    ds_val = load_foodseg103("validation")

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

#TODO finish this
class FoodSegDataset(PyTorchDataset):
    """
    Dataset for FoodSeg103 for fine-tuning SegGPT using the Hugging Face datasets library.
    """
    def __init__(
        self, 
        config: SegGptConfig,
        image_processor: SegGptImageProcessor, 
        split: str = "train",
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
    

if __name__ == "__main__":
    config = SegGptConfig.from_pretrained("BAAI/seggpt-vit-large")
    image_processor = SegGptImageProcessor.from_pretrained("BAAI/seggpt-vit-large")
    transform_train = TransformInContextTuning(config, image_processor)

    ds_train = load_foodseg103("train")
    ds_train.set_transform(transform_train)
    sample = ds_train[0]

    print(sample)