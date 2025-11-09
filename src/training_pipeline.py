import pandas as pd
from pathlib import Path
import sys

# Get project root from the script's location
# This file is in src/, so project root is the parent directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import TanaForecast
from src.trainer import TanaForecastTrainer, TimeSeriesDataset
from src.utils import Loss
from typing import List, Optional

CONTEXT_WINDOW = 4096
PREDICTION_LENGTH = 256

class RunDatasetTraining:
    def __init__(
        self,
        dataset_name: str,
        train_dataset: pd.DataFrame,
        test_dataset: Optional[pd.DataFrame],
        context_window: int,
        prediction_length: int,
        feature_columns: List[str],
        target_columns: List[str],
        timestamp_column: str
    ) -> None:
        self.dataset_name = dataset_name
        self.context_window = context_window
        self.prediction_length = prediction_length
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.timestamp_column = timestamp_column
        
        # Store original DataFrames
        train_df = train_dataset
        test_df = test_dataset
        
        # Create TimeSeriesDataset instances
        self.train_dataset = TimeSeriesDataset(
            df=train_df,
            context_window=self.context_window,
            prediction_length=self.prediction_length,
            stride=1,
            normalize=True,
            feature_columns=self.feature_columns,
            target_columns=self.target_columns,
            timestamp_column=self.timestamp_column
        )

        self.test_dataset = TimeSeriesDataset(
            df=test_df,
            context_window=self.context_window,
            prediction_length=self.prediction_length,
            stride=1,
            normalize=True,
            feature_columns=self.feature_columns,
            target_columns=self.target_columns,
            timestamp_column=self.timestamp_column
        ) if test_df is not None else None

        self.model = TanaForecast(
            context_window=self.context_window,
            prediction_length=self.prediction_length
        )

        self.trainer = TanaForecastTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
            batch_size=64,
            learning_rate=1e-3,
            dataset_name=self.dataset_name,
            num_epochs=10,
            checkpoint_dir=str(project_root / 'checkpoints'),
            early_stopping_patience=-1,
            loss_fn=Loss.quantile_loss,
            loss_name='Quantile_0.9',
            loss_kwargs={'q': 0.9}
        )

    def run(self) -> None:
        self.trainer.train()

if __name__ == "__main__":
    train_dataset = pd.read_csv(project_root / 'src' / 'datasets' / 'stock_china' / '000001.XSHE.csv')

    feature_columns = ['open','high','low','volume','money','avg','high_limit','low_limit','pre_close','paused','factor']
    target_columns = ['close']
    timestamp_column = 'timestamp'

    run_dataset_training = RunDatasetTraining(
        train_dataset=train_dataset,
        dataset_name='microsoft_stocks',
        test_dataset=None,
        context_window=CONTEXT_WINDOW,
        prediction_length=PREDICTION_LENGTH,
        feature_columns=feature_columns,
        target_columns=target_columns,
        timestamp_column=timestamp_column
    )
    
    run_dataset_training.run()