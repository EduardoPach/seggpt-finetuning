import os

from accelerate import Accelerator
from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    AutoImageProcessor,
    SegGptForImageSegmentation
)

from data import get_fine_tuning_loaders, DataTrainingArguments

def train_step():
    ...


def main() -> None:
    os.environ["WANDB_PROJECT"] = "seggpt-fine-tuning"

    

    accelerator = Accelerator()

if __name__ == "__main__":
    main()