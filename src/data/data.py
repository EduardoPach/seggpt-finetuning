import math
import itertools
from typing import List, Tuple, Dict, Optional

import torch
from datasets import Dataset
from transformers import SegGptImageProcessor, SegGptConfig

from data.utils import (
    NUM_CLASSES,
    load_foodseg103,
    random_masking,
    random_color_palette,
    mask_coloring,
    collate_fn,
    DataTrainingArguments
)

class FoodSeg103InContextDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        dataset: Dataset,
        config: SegGptConfig,
        image_processor: SegGptImageProcessor,
        mask_ratio: float = 0.75,
        random_coloring: bool = True,
        is_train: bool = True
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.image_processor = image_processor
        self.mask_ratio = mask_ratio
        self.random_coloring = random_coloring
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        num_patches = math.prod(i // self.config.patch_size for i in self.config.image_size)

        sample = self.dataset[idx]

        image = sample["image"]
        label = sample["label"]

        bool_masked_pos = None
        if self.is_train:
            bool_masked_pos = random_masking(num_patches, mask_ratio=self.mask_ratio)

        if self.random_coloring:
            palette = random_color_palette(list(range(1, NUM_CLASSES)))
            label = mask_coloring(label, palette)

        inputs = self.image_processor(
            images=image, 
            prompt_masks=mask_coloring(label) if self.random_coloring else label, 
            num_labels=NUM_CLASSES - 1, 
            do_convert_rgb=not self.random_coloring,
            return_tensors="pt"
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": inputs["prompt_masks"].squeeze(0),
            "bool_masked_pos": bool_masked_pos
        }


class FoodSeg103Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        dataset: Dataset,
        config: SegGptConfig,
        image_processor: SegGptImageProcessor,
        mask_ratio: float = 0.75,
        random_coloring: bool = True,
        num_pairs_per_image: Optional[int] = None,
        is_train: bool = True
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.image_processor = image_processor
        self.mask_ratio = mask_ratio
        self.random_coloring = random_coloring
        self.is_train = is_train
        self.num_pairs_per_image = num_pairs_per_image

        self.pairs = self.get_pairs()


    def get_pairs(self) -> List[Tuple[int, int]]:
        classes = [set(cls) for cls in self.dataset["classes_on_image"]]
        ids = self.dataset["id"]

        pairs = itertools.combinations(ids, 2)

        # To be a valid pair should have at least one class in common
        pairs = [(i, j) for i, j in pairs if len(classes[i].intersection(classes[j])) > 0]
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        num_patches = math.prod(i // self.config.patch_size for i in self.config.image_size)

        input_idx, prompt_idx = self.pairs[idx]

        input_sample = self.dataset[input_idx]
        prompt_sample = self.dataset[prompt_idx]

        image = input_sample["image"]
        label = input_sample["label"]
        prompt_image = prompt_sample["image"]
        prompt_mask = prompt_sample["label"]

        bool_masked_pos = None
        if self.is_train:
            bool_masked_pos = random_masking(num_patches, mask_ratio=self.mask_ratio)


        if self.random_coloring:
            palette = random_color_palette(list(range(1, NUM_CLASSES)))
            prompt_mask = mask_coloring(prompt_mask, palette)
            label = mask_coloring(label, palette)

        inputs = self.image_processor(
            images=image, 
            prompt_masks=prompt_mask,
            prompt_images=prompt_image,
            num_labels=NUM_CLASSES - 1,
            do_convert_rgb=not self.random_coloring, 
            return_tensors="pt"
        )

        labels_inputs = self.image_processor(
            images=None,
            prompt_masks=label,
            num_labels=NUM_CLASSES - 1,
            do_convert_rgb=not self.random_coloring,
            return_tensors="pt"
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "prompt_pixel_values": inputs["prompt_pixel_values"].squeeze(0),
            "prompt_masks": inputs["prompt_masks"].squeeze(0),
            "labels": labels_inputs["prompt_masks"].squeeze(0),
            "bool_masked_pos": bool_masked_pos
        }

def get_in_context_datasets(
    config: SegGptConfig, 
    image_processor: SegGptImageProcessor, 
    data_args: DataTrainingArguments
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Returns the training and validation datasets for the FoodSeg103 dataset for in-context training."""
    ds_train = load_foodseg103("train")
    ds_val = load_foodseg103("validation")

    train_dataset = FoodSeg103InContextDataset(
        ds_train, 
        config, 
        image_processor, 
        data_args.mask_ratio, 
        data_args.random_coloring, 
        is_train=True
    )
    validation_dataset = FoodSeg103InContextDataset(
        ds_val, 
        config, 
        image_processor, 
        data_args.mask_ratio, 
        data_args.random_coloring, 
        is_train=False
    )

    return train_dataset, validation_dataset

def get_fine_tuning_datasets(
    config: SegGptConfig,
    image_processor: SegGptImageProcessor,
    data_args: DataTrainingArguments,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Returns the training and validation datasets for the FoodSeg103 for full image fine-tuning."""
    ds_train = load_foodseg103("train")
    ds_val = load_foodseg103("validation")

    train_dataset = FoodSeg103Dataset(
        ds_train, 
        config, 
        image_processor, 
        data_args.mask_ratio, 
        data_args.random_coloring,
        data_args.num_pairs_per_image,
        is_train=True
    )

    validation_dataset = FoodSeg103Dataset(
        ds_val, 
        config, 
        image_processor, 
        data_args.mask_ratio, 
        data_args.random_coloring,
        data_args.num_pairs_per_image,
        is_train=False
    )

    return train_dataset, validation_dataset

def get_fine_tuning_loaders(
    config: SegGptConfig,
    image_processor: SegGptImageProcessor,
    data_args: DataTrainingArguments
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Returns the training and validation datasets for the FoodSeg103 for full image fine-tuning."""
    train_dataset, val_dataset = get_fine_tuning_datasets(
        config, 
        image_processor, 
        data_args
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=data_args.train_batch_size,
        num_workers=data_args.num_workers,
        pin_memory=data_args.pin_memory,
        shuffle=True, 
        collate_fn=collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=data_args.validation_batch_size, 
        num_workers=data_args.num_workers,
        pin_memory=True,
        shuffle=False, 
        collate_fn=collate_fn
    )

    return train_loader, val_loader

def get_in_context_loaders(
    config: SegGptConfig,
    image_processor: SegGptImageProcessor,
    data_args: DataTrainingArguments
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Returns the training and validation datasets for the FoodSeg103 for in-context training."""
    train_dataset, val_dataset = get_in_context_datasets(
        config, 
        image_processor, 
        data_args
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=data_args.train_batch_size,
        num_workers=data_args.num_workers,
        pin_memory=data_args.pin_memory,
        shuffle=True, 
        collate_fn=collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=data_args.validation_batch_size, 
        num_workers=data_args.num_workers,
        pin_memory=data_args.pin_memory,
        shuffle=False, 
        collate_fn=collate_fn
    )

    return train_loader, val_loader
    
if __name__ == "__main__":
    ds_train = load_foodseg103("train")
    train_dummy = load_foodseg103("train") # Need the dummy otherwise will recurse infinitely

    train_dataset, _ = get_fine_tuning_datasets(
        SegGptConfig.from_pretrained("BAAI/seggpt-vit-large"), 
        SegGptImageProcessor.from_pretrained("BAAI/seggpt-vit-large"),
        mask_ratio=0.75,
        random_coloring=True,
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)


    for inputs in train_loader:
        print(inputs)
        print("here")
