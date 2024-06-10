import os
import sys

from transformers import Trainer, AutoImageProcessor

from src.train import SegGptAdapter, get_terminal_args, get_config_args
from src.data import get_in_context_datasets, DataTrainingArguments


def main() -> None:
    os.environ["WANDB_PROJECT"] = "seggpt-incontext-tuning"
    config_path = os.path.join(os.path.dirname(__file__), "config", "config_adapter.yaml")

    args = get_terminal_args()
    training_args, data_args = get_config_args(args, config_path)

    model_id = "BAAI/seggpt-vit-large"
    model = SegGptAdapter(model_id)
    config = model.seggpt.config
    image_processor = AutoImageProcessor.from_pretrained(model_id)

    train_dataset, validation_dataset = get_in_context_datasets(
        config,
        image_processor,
        data_args
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    trainer.train()

if __name__ == "__main__":
    main()