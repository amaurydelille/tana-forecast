import pandas as pd
from pathlib import Path
import sys

# Get project root from the script's location
# This file is in src/, so project root is the parent directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import TanaForecast
from src.trainer import TanaForecastTrainer, TimeSeriesDataset
from src.utils import Loss, Constants
from typing import List, Optional

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
        self.original_context_window = context_window
        self.prediction_length = prediction_length
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.timestamp_column = timestamp_column
        
        # Determine train/test DataFrames
        if test_dataset is not None:
            train_df = train_dataset
            test_df = test_dataset
        else:
            split_index = int(len(train_dataset) * 0.8)
            train_df = train_dataset.iloc[:split_index].copy()
            test_df = train_dataset.iloc[split_index:].copy()

        self.context_window = self._get_effective_context_window(
            df=train_df,
            requested_context_window=context_window,
            split_name="train"
        )

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

        self.test_dataset = self._build_optional_dataset(
            df=test_df,
            split_name="test"
        )

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

    def _get_effective_context_window(
        self,
        df: pd.DataFrame,
        requested_context_window: int,
        split_name: str
    ) -> int:
        total_rows = len(df)
        max_context = total_rows - self.prediction_length

        if max_context <= 0:
            raise ValueError(
                f"{split_name.capitalize()} dataset for '{self.dataset_name}' "
                f"needs at least {self.prediction_length + 1} rows to create "
                f"a single training sample (found {total_rows})."
            )

        effective_context = min(requested_context_window, max_context)
        if effective_context < requested_context_window:
            print(
                f"[{self.dataset_name}] Adjusted {split_name} context window "
                f"from {requested_context_window} to {effective_context} to fit the available history."
            )

        return effective_context

    def _build_optional_dataset(
        self,
        df: Optional[pd.DataFrame],
        split_name: str
    ) -> Optional[TimeSeriesDataset]:
        if df is None:
            return None

        total_rows = len(df)
        min_required = self.context_window + self.prediction_length

        if total_rows < min_required:
            print(
                f"[{self.dataset_name}] Skipping {split_name} dataset: "
                f"requires at least {min_required} rows (found {total_rows})."
            )
            return None

        return TimeSeriesDataset(
            df=df,
            context_window=self.context_window,
            prediction_length=self.prediction_length,
            stride=1,
            normalize=True,
            feature_columns=self.feature_columns,
            target_columns=self.target_columns,
            timestamp_column=self.timestamp_column
        )

if __name__ == "__main__":
    train_dataset = pd.read_csv(project_root / 'src' / 'datasets' / 'stock_china' / '000001.XSHE.csv')
    print(train_dataset.shape)
    LAST_INDEX = 1000
    train_dataset = train_dataset.iloc[:LAST_INDEX]
    feature_columns = ['open','high','low','volume', 'money', 'avg', 'high_limit', 'low_limit', 'pre_close', 'paused', 'factor']
    target_columns = ['close']
    timestamp_column = 'timestamp'

    run_dataset_training = RunDatasetTraining(
        train_dataset=train_dataset.copy(),
        dataset_name='stock_china',
        test_dataset=None,
        context_window=Constants.context_window,
        prediction_length=Constants.prediction_length,
        feature_columns=feature_columns,
        target_columns=target_columns,
        timestamp_column=timestamp_column
    )
    
    run_dataset_training.run()