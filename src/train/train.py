import os

from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    AutoImageProcessor,
    SegGptForImageSegmentation
)

from src.data import get_fine_tuning_datasets, DataTrainingArguments, collate_fn


def main() -> None:
    os.environ["WANDB_PROJECT"] = "seggpt-fine-tuning"
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.yaml")

    parser = HfArgumentParser((DataTrainingArguments, TrainingArguments))
    data_args, training_args = parser.parse_yaml_file(CONFIG_PATH)

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