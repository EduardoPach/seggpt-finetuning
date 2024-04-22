import json
import itertools
from typing import List, Tuple, Dict, Any

import torch
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download
from transformers import SegGptImageProcessor, SegGptConfig

DATASET_ID = "EduardoPacheco/FoodSeg103"

class FoodSegLabelMapper:
    def __init__(self) -> None:
        id2label = json.load(open(hf_hub_download(DATASET_ID, "id2label.json", repo_type="dataset"), "r"))
        self.id2label = {int(k): v for k, v in id2label.items()}
        self.label2id = {v: k for k, v in id2label.items()}

    def __len__(self) -> int:
        return len(self.id2label)
    
    def get_label(self, idx: int) -> str:
        return self.id2label[idx]
    
    def get_id(self, label: str) -> int:
        return self.label2id[label]

#TODO finish this
class FoodSegDataset(Dataset):
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