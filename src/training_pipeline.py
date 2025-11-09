import pandas as pd
from pathlib import Path
import sys

project_root = Path.cwd().parent.parent
sys.path.insert(0, str(project_root))

from src.model import TanaForecast
from src.trainer import TanaForecastTrainer, TimeSeriesDataset, Logger
from src.utils import Loss
from typing import List

CONTEXT_WINDOW = 90
PREDICTION_LENGTH = 7

class RunDatasetTraining:
    def __init__(
        self,
        dataset_name: str,
        train_dataset: pd.DataFrame,
        test_dataset: pd.DataFrame,
        context_window: int,
        prediction_length: int,
        feature_columns: List[str],
        target_columns: List[str]
    ) -> None:
        self.dataset_name = dataset_name
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.context_window = context_window
        self.prediction_length = prediction_length

        self.train_dataset = TimeSeriesDataset(
            df=self.train_dataset,
            context_window=self.context_window,
            prediction_length=self.prediction_length,
            stride=1,
            normalize=True,
            feature_columns=self.feature_columns,
            target_columns=self.target_columns
        )

        self.test_dataset = TimeSeriesDataset(
            df=self.test_dataset,
            context_window=self.context_window,
            prediction_length=self.prediction_length,
            stride=1,
            normalize=True,
            feature_columns=self.feature_columns,
            target_columns=self.target_columns
        )

        self.model = TanaForecast(
            context_window=self.context_window,
            prediction_length=self.prediction_length
        )

        trainer = TanaForecastTrainer(
            model=self.model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=64,
            learning_rate=1e-3,
            dataset_name=self.dataset_name,
            num_epochs=3,
            checkpoint_dir=str(project_root / 'checkpoints' / 'training'),
            early_stopping_patience=-1,
            loss_fn=Loss.quantile_loss,
            loss_name='Quantile_0.9',
            loss_kwargs={'q': 0.9}
        )

    def run(self) -> None:
        self.trainer.train()

if __name__ == "__main__":
    train_dataset = pd.read_csv(project_root / 'src' / 'datasets' / 'delhi' / 'train.csv')
    test_dataset = pd.read_csv(project_root / 'src' / 'datasets' / 'delhi' / 'test.csv')

    feature_columns = ['humidity','wind_speed','meanpressure']
    target_columns = ['meantemp']

    run_dataset_training = RunDatasetTraining(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        context_window=CONTEXT_WINDOW,
        prediction_length=PREDICTION_LENGTH,
        feature_columns=feature_columns,
        target_columns=target_columns
    )