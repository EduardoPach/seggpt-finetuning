import os

from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    AutoImageProcessor,
    SegGptForImageSegmentation
)

from data import get_fine_tuning_datasets, DataTrainingArguments

def train_seggpt_adapter(training_args: TrainingArguments, data_args: DataTrainingArguments) -> None:
    """Trains the SegGptAdapter model."""
    model_id = "BAAI/seggpt-vit-large"

    model = SegGptForImageSegmentation.from_pretrained(model_id)
    image_processor = AutoImageProcessor.from_pretrained(model_id)

    train_dataset, validation_dataset = get_fine_tuning_datasets(
        model.config,
        image_processor,
        data_args.mask_ratio,
        data_args.random_coloring
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    trainer.train()

def main() -> None:
    os.environ["WANDB_PROJECT"] = "seggpt-fine-tuning"

    parser = HfArgumentParser((TrainingArguments, DataTrainingArguments))
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    training_args, data_args = parser.parse_yaml_file(yaml_file=config_path)

    train_seggpt_adapter(training_args, data_args)

if __name__ == "__main__":
    main()