import os
import sys
from dataclasses import dataclass, field

from datasets import Dataset
from transformers import Trainer, TrainingArguments, AutoImageProcessor, HfArgumentParser

from seggpt_adapter import SegGptAdapter
from data import get_datasets, FoodSegLabelMapper


@dataclass
class DataTrainingArguments:
    mask_ratio: float = field(default=0.75)
    random_coloring: bool = field(default=True)


def train_seggpt_adapter(training_args: TrainingArguments, data_args: DataTrainingArguments) -> None:
    """Trains the SegGptAdapter model."""
    model_id = "BAAI/seggpt-vit-large"

    model = SegGptAdapter(model_id)
    config = model.seggpt.config
    image_processor = AutoImageProcessor.from_pretrained(model_id)

    train_dataset, validation_dataset = get_datasets(
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
    )

    trainer.train()

def main() -> None:
    os.environ["WANDB_PROJECT"] = "seggpt-incontext-tuning"

    parser = HfArgumentParser((TrainingArguments, DataTrainingArguments))
    path = "src/config/config_adapter.yaml"
    training_args, data_args = parser.parse_yaml_file(yaml_file=os.path.abspath(path))
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
    #     training_args, data_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    # else:
    #     training_args, data_args = parser.parse_args_into_dataclasses()

    train_seggpt_adapter(training_args, data_args)

if __name__ == "__main__":
    main()