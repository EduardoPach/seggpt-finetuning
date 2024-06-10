import os
from pathlib import Path

from transformers import (
    Trainer,
    AutoImageProcessor,
    SegGptForImageSegmentation
)

from src.train import get_config_args, get_terminal_args
from src.data import get_fine_tuning_datasets, collate_fn


def main() -> None:
    os.environ["WANDB_PROJECT"] = "seggpt-fine-tuning"
    config_path = str(Path(__file__).resolve().parent.parent / "config" / "config.yaml")
    print("using config path: ", config_path)

    args = get_terminal_args()
    training_args, data_args = get_config_args(args, config_path)

    model = SegGptForImageSegmentation.from_pretrained("BAAI/seggpt-vit-large")
    image_processor = AutoImageProcessor.from_pretrained("BAAI/seggpt-vit-large")
    config = model.config

    train_dataset, val_dataset = get_fine_tuning_datasets(
        config, 
        image_processor, 
        data_args
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn
    )

    trainer.train()

if __name__ == "__main__":
    main()