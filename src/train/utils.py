import argparse
from typing import Tuple

from datasets import Dataset
from transformers.integrations import WandbCallback
from transformers import Trainer, SegGptImageProcessor, TrainingArguments, HfArgumentParser

from src.data import DataTrainingArguments

class KwargsAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Initialize or fetch the kwargs dictionary
        if not hasattr(namespace, 'kwargs'):
            setattr(namespace, 'kwargs', {})
        kwargs = getattr(namespace, 'kwargs')

        # Parse the key-value pair
        for value in values:
            key, val = value.split('=')
            kwargs[key] = val

        setattr(namespace, 'kwargs', kwargs)


class WandbReconstructionCallback(WandbCallback):
    def __init__(
        self,
        trainer: Trainer,
        validation_dataset: Dataset,
        image_processor: SegGptImageProcessor,
        num_samples: int = 1,
        freq: int = 1
    ) -> None:
        super().__inti__()
        self.trainer = trainer
        self.validation_dataset = validation_dataset
        self.image_processor = image_processor
        self.num_samples = num_samples
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        if state.epoch % self.freq == 0:
            ...

def get_terminal_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kwargs", action=KwargsAction, nargs='*')
    return parser.parse_args()

def get_config_args(args: argparse.Namespace, config_path: str) -> Tuple[TrainingArguments, DataTrainingArguments]:
    parser = HfArgumentParser((TrainingArguments, DataTrainingArguments))
    training_args, data_args = parser.parse_yaml_file(config_path)

    kwargs = args.kwargs

    for key, val in kwargs.items():
        if hasattr(training_args, key):
            setattr(training_args, key, val)
        elif hasattr(data_args, key):
            setattr(data_args, key, val)
        else:
            raise ValueError(f"Invalid argument: {key}")
        
    return training_args, data_args