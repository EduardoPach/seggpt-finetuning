import math
import random
import itertools
from collections import defaultdict
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
        is_train: bool = True,
        num_samples: Optional[int] = None,
        **kwargs
    ) -> None:
        self.dataset = dataset if num_samples is None else dataset.select(range(num_samples))
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
        is_train: bool = True,
        seed: Optional[int] = None,
        num_pairs_per_image: Optional[int] = None,
        num_samples: Optional[int] = None
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.image_processor = image_processor
        self.mask_ratio = mask_ratio
        self.random_coloring = random_coloring
        self.is_train = is_train
        self.seed = seed
        self.num_pairs_per_image = num_pairs_per_image if num_pairs_per_image is not None else float("inf")
        self.num_samples = num_samples

        self.pairs = self.get_pairs()


    def get_pairs(self) -> List[Tuple[int, int]]:
        classes = [set(cls) for cls in self.dataset["classes_on_image"]]
        ids = self.dataset["id"]

        # To be a valid pair should have at least one class in common
        valid_pairs = lambda x, y: len(classes[x].intersection(classes[y])) > 0

        pairs = []
        pairs_per_id = defaultdict(int)

        for i, j in itertools.combinations(ids, 2):
            if not valid_pairs(i, j):
                continue

            if pairs_per_id[i] < self.num_pairs_per_image:
                pairs.append((i, j))
                pairs_per_id[i] += 1

            if pairs_per_id[j] < self.num_pairs_per_image:
                pairs.append((j, i))
                pairs_per_id[j] += 1

        if self.num_samples is not None:
            if self.seed is not None: random.seed(self.seed)

            start_idx = random.randint(0, len(pairs) - self.num_samples)
            end_idx = start_idx + self.num_samples
            pairs = pairs[start_idx:end_idx]

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
        dataset=ds_train, 
        config=config, 
        image_processor=image_processor, 
        mask_ratio=data_args.mask_ratio, 
        random_coloring=data_args.random_coloring, 
        num_samples=data_args.num_samples,
        is_train=True
    )
    validation_dataset = FoodSeg103InContextDataset(
        dataset=ds_val, 
        config=config, 
        image_processor=image_processor, 
        mask_ratio=data_args.mask_ratio, 
        random_coloring=data_args.random_coloring, 
        num_samples=data_args.num_samples,
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
        dataset=ds_train, 
        config=config, 
        image_processor=image_processor, 
        mask_ratio=data_args.mask_ratio, 
        random_coloring=data_args.random_coloring,
        num_pairs_per_image=data_args.num_pairs_per_image,
        num_samples=data_args.num_samples,
        is_train=True
    )

    validation_dataset = FoodSeg103Dataset(
        dataset=ds_val, 
        config=config, 
        image_processor=image_processor, 
        mask_ratio=data_args.mask_ratio, 
        random_coloring=data_args.random_coloring,
        num_pairs_per_image=data_args.num_pairs_per_image,
        num_samples=data_args.num_samples,
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
