import os
import sys

from transformers import Trainer, TrainingArguments, AutoImageProcessor, HfArgumentParser

from seggpt_adapter import SegGptAdapter
from data import get_in_context_datasets, collate_fn, DataTrainingArguments

def train_seggpt_adapter(training_args: TrainingArguments, data_args: DataTrainingArguments) -> None:
    """Trains the SegGptAdapter model."""
    model_id = "BAAI/seggpt-vit-large"

    model = SegGptAdapter(model_id)
    config = model.seggpt.config
    image_processor = AutoImageProcessor.from_pretrained(model_id)

    train_dataset, validation_dataset = get_in_context_datasets(
        config,
        image_processor,
        data_args.mask_ratio,
        data_args.random_coloring
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collate_fn,
    )

    trainer.train()

def main() -> None:
    os.environ["WANDB_PROJECT"] = "seggpt-incontext-tuning"

    parser = HfArgumentParser((TrainingArguments, DataTrainingArguments))
    config_path = os.path.join(os.path.dirname(__file__), "config", "config_adapter.yaml")
    training_args, data_args = parser.parse_yaml_file(yaml_file=config_path)

    train_seggpt_adapter(training_args, data_args)

if __name__ == "__main__":
    main()