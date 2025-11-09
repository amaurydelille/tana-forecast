import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import time
import csv
from src.utils import TimeStamps
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Logger:
    """Log training runs into CSV files"""
    def __init__(self, csv_path: str) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = ["run_type", "dataset_name", "dataset_shape", "dataset_size", "model_parameters", "context_window", "prediction_length", "epoch", "total_epochs", "loss_type", "training_loss", "validation_loss", "training_history", "validation_history", "timestamp"]
        
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        self.csv_file = open(self.csv_path, "a", newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.columns)
        
        if not file_exists:
            self.csv_writer.writeheader()
            self.csv_file.flush()

    def log(self, run_type: str = None, dataset_name: str = None, dataset_shape: Tuple[int, int, int] = None, dataset_size: int = None, model_parameters: int = None, context_window: int = None, prediction_length: int = None, epoch: int = None, total_epochs: int = None, loss_type: str = None, training_loss: float = None, validation_loss: float = None, training_history: List[float] = None, validation_history: List[float] = None, timestamp: float = None) -> None:
        self.csv_writer.writerow({
            "run_type": run_type,
            "dataset_name": dataset_name,
            "dataset_shape": str(dataset_shape),
            "dataset_size": dataset_size,
            "model_parameters": model_parameters,
            "context_window": context_window,
            "prediction_length": prediction_length,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "loss_type": loss_type,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "training_history": str(training_history),
            "validation_history": str(validation_history),
            "timestamp": timestamp if timestamp is not None else time.time()
        })
        self.csv_file.flush()
    
    def close(self) -> None:
        """Close the CSV file"""
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()
    
    def __del__(self) -> None:
        """Ensure file is closed when Logger is destroyed"""
        self.close()

class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        context_window: int,
        prediction_length: int,
        target_columns: Optional[List[str]] = None,
        feature_columns: Optional[List[str]] = None,
        timestamp_column: Optional[str] = None,
        stride: int = 1,
        normalize: bool = True,
        flatten: bool = True
    ) -> None:
        if hasattr(df, 'to_pandas'):
            self.df = df.to_pandas()
        elif isinstance(df, pd.DataFrame):
            self.df = df.copy()
        
        if flatten:
            self.df = self._flatten_dataframe(self.df)
        
        self.context_window = context_window
        self.prediction_length = prediction_length
        self.stride = stride
        
        if feature_columns is None:
            self.feature_columns = list(self.df.columns)
        else:
            self.feature_columns = feature_columns
            
        if target_columns is None:
            self.target_columns = self.feature_columns
        else:
            self.target_columns = target_columns
        
        self.df = self._normalize_timestamp_columns(self.df, self.feature_columns, timestamp_column)
        
        self.feature_data = self.df[self.feature_columns].values.astype(np.float32)
        self.target_data = self.df[self.target_columns].values.astype(np.float32)
        
        if normalize:
            self.feature_mean = self.feature_data.mean(axis=0)
            self.feature_std = self.feature_data.std(axis=0) + 1e-8
            self.feature_data = (self.feature_data - self.feature_mean) / self.feature_std
            
            self.target_mean = self.target_data.mean(axis=0)
            self.target_std = self.target_data.std(axis=0) + 1e-8
            self.target_data = (self.target_data - self.target_mean) / self.target_std
        else:
            self.feature_mean = None
            self.feature_std = None
            self.target_mean = None
            self.target_std = None
        
        self.valid_indices = self._compute_valid_indices()
    
    @staticmethod
    def _flatten_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        flattened_dfs = []
        needs_flattening = False
        
        for index, row in df.iterrows():
            first_value = row.iloc[0]
            if isinstance(first_value, (list, np.ndarray)) and len(first_value) > 1:
                needs_flattening = True
                flat_data = {}
                for col in df.columns:
                    flat_data[col] = row[col]
                flattened_dfs.append(pd.DataFrame(flat_data))
            else:
                flattened_dfs.append(pd.DataFrame([row]))
        
        if needs_flattening and flattened_dfs:
            return pd.concat(flattened_dfs, ignore_index=True)
        return df
    
    @staticmethod
    def _normalize_timestamp_columns(df: pd.DataFrame, feature_columns: List[str], timestamp_column: Optional[str] = None) -> pd.DataFrame:
        """
        Normalize timestamp columns in the dataframe. Only normalizes the specified timestamp_column.
        
        Handles:
        - pandas datetime columns
        - string timestamps
        - torch.Tensor timestamps (from Chronos datasets)
        - numpy array timestamps
        
        Args:
            df: Input dataframe
            feature_columns: List of feature column names (not used when timestamp_column is specified)
            timestamp_column: Name of the timestamp column to normalize. If None, no normalization is performed.
        
        Returns:
            DataFrame with timestamp column normalized to numeric values (days since 1970-01-01)
        """
        df = df.copy()
        
        # Only normalize the specified timestamp_column
        if timestamp_column is None:
            return df
        
        if timestamp_column not in df.columns:
            return df
        
        col_data = df[timestamp_column]
                
        if isinstance(col_data.iloc[0] if len(col_data) > 0 else None, torch.Tensor):
            tensor_timestamps = torch.stack([x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in col_data.values])
            normalized = TimeStamps.universal_timestamp_normalized(tensor_timestamps, device="cpu")
            df[timestamp_column] = normalized.numpy()
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            normalized = TimeStamps.universal_timestamp_normalized(col_data, device="cpu")
            df[timestamp_column] = normalized.numpy()
        elif col_data.dtype == 'object':
            try:
                test_val = col_data.iloc[0] if len(col_data) > 0 else None
                if test_val is not None:
                    if isinstance(test_val, (torch.Tensor, np.ndarray)):
                        normalized = TimeStamps.universal_timestamp_normalized(col_data.values, device="cpu")
                        df[timestamp_column] = normalized.numpy()
                    else:
                        pd.to_datetime(col_data.iloc[:5])
                        normalized = TimeStamps.universal_timestamp_normalized(col_data, device="cpu")
                        df[timestamp_column] = normalized.numpy()
            except (ValueError, TypeError, AttributeError):
                pass
        
        return df
    
    def _compute_valid_indices(self) -> List[int]:
        max_start_idx = len(self.df) - self.context_window - self.prediction_length
        return list(range(0, max_start_idx + 1, self.stride))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.context_window
        target_end_idx = end_idx + self.prediction_length
        
        context = self.feature_data[start_idx:end_idx]
        target = self.target_data[end_idx:target_end_idx]
        
        context_tensor = torch.from_numpy(context).transpose(0, 1)
        target_tensor = torch.from_numpy(target).transpose(0, 1)
        
        return context_tensor, target_tensor
    
    def denormalize(self, tensor: torch.Tensor, is_target: bool = True) -> torch.Tensor:
        if is_target and self.target_mean is not None:
            mean = torch.from_numpy(self.target_mean).to(tensor.device)
            std = torch.from_numpy(self.target_std).to(tensor.device)
        elif not is_target and self.feature_mean is not None:
            mean = torch.from_numpy(self.feature_mean).to(tensor.device)
            std = torch.from_numpy(self.feature_std).to(tensor.device)
        else:
            return tensor
        
        if tensor.dim() > 1:
            mean = mean.view(-1, 1)
            std = std.view(-1, 1)
        
        return tensor * std + mean

class TanaForecastTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: TimeSeriesDataset,
        test_dataset: Optional[TimeSeriesDataset] = None,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        num_epochs: int = 100,
        device: str = "mps" if torch.backends.mps.is_available() else "cpu",
        checkpoint_dir: Optional[str] = None,
        early_stopping_patience: int = -1,
        dataset_name: str = "unknown",
        loss_fn: Optional[callable] = None,
        loss_name: str = "MSE",
        loss_kwargs: Optional[Dict] = None,
        allow_rerun: bool = False
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.logger = Logger(csv_path='/Users/amaurydelille/Documents/projects/tana-forecast/src/logs/training_logs.csv')
        self.dataset_name = dataset_name

        logs = pd.read_csv(self.logger.csv_path)
        if dataset_name in logs['dataset_name'].values and not allow_rerun:
            raise ValueError(f"Dataset {dataset_name} already exists in logs. Set allow_rerun=True to overwrite.")

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.allow_rerun = allow_rerun
        self.loss_name = loss_name
        self.loss_kwargs = loss_kwargs or {}
        if loss_fn is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = loss_fn
        self.use_custom_loss = loss_fn is not None
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if device == "mps" else False
        )
        
        if test_dataset is not None:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
        else:
            self.test_loader = None
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=learning_rate * 0.01
        )
        
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'learning_rates': []
        }
        
        self.best_test_loss = float('inf')
        self.epochs_without_improvement = 0

    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count the total number of trainable parameters in a model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def get_dataset_shape(self) -> Tuple[int, int, int]:
        """
        Get the shape of the dataset.
        Returns: (num_samples, num_features, context_window)
        """
        num_samples = len(self.train_dataset)
        context, _ = self.train_dataset[0]
        num_features, context_window = context.shape
        return (num_samples, num_features, context_window)
        
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for context, target in self.train_loader:
            context = context.to(self.device)
            target = target.to(self.device)
            
            if target.dim() == 3 and target.size(1) == 1:
                target = target.squeeze(1)
            
            self.optimizer.zero_grad()
            
            if hasattr(self.model, 'return_router_info') and self.model.return_router_info:
                predictions, router_probs, expert_indices = self.model(context)
                loss = self.criterion(
                    y_true=target,
                    y_pred=predictions,
                    router_probs=router_probs,
                    expert_indices=expert_indices,
                    **self.loss_kwargs
                )
            else:
                predictions = self.model(context)
                if self.use_custom_loss:
                    loss = self.criterion(target, predictions, **self.loss_kwargs)
                else:
                    loss = self.criterion(predictions, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            logger.info(f"Train batch {num_batches} loss: {loss.item()} in {time.time() - start_time:.2f}s")
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        if self.test_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for context, target in self.test_loader:
                context = context.to(self.device)
                target = target.to(self.device)
                
                if target.dim() == 3 and target.size(1) == 1:
                    target = target.squeeze(1)
                
                if hasattr(self.model, 'return_router_info') and self.model.return_router_info:
                    predictions, router_probs, expert_indices = self.model(context)
                    loss = self.criterion(
                        y_true=target,
                        y_pred=predictions,
                        router_probs=router_probs,
                        expert_indices=expert_indices,
                        **self.loss_kwargs
                    )
                else:
                    predictions = self.model(context)
                    if self.use_custom_loss:
                        loss = self.criterion(target, predictions, **self.loss_kwargs)
                    else:
                        loss = self.criterion(predictions, target)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint at the end of training."""
        if self.checkpoint_dir is None:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_test_loss': self.best_test_loss,
            'history': self.history
        }
        
        path = self.checkpoint_dir / 'final_model.pt'
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # Handle both 'best_val_loss' and 'best_test_loss' for backwards compatibility
        if 'best_test_loss' in checkpoint:
            self.best_test_loss = checkpoint['best_test_loss']
        elif 'best_val_loss' in checkpoint:
            self.best_test_loss = checkpoint['best_val_loss']
        else:
            self.best_test_loss = float('inf')
        # Handle history keys for backwards compatibility
        if 'history' in checkpoint:
            history = checkpoint['history']
            # Convert 'val_loss' to 'test_loss' if needed
            if 'val_loss' in history and 'test_loss' not in history:
                history['test_loss'] = history.pop('val_loss')
            self.history = history
        else:
            self.history = {
                'train_loss': [],
                'test_loss': [],
                'learning_rates': []
            }
        return checkpoint['epoch']
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint (final_model.pt)."""
        if self.checkpoint_dir is None or not self.checkpoint_dir.exists():
            return None
        
        final_model_path = self.checkpoint_dir / 'final_model.pt'
        if final_model_path.exists():
            return str(final_model_path)
        
        return None
    
    def load_latest_checkpoint(self) -> int:
        """Load the latest checkpoint (final_model.pt) if it exists.
        
        Returns:
            The next epoch to start training from (0 if no checkpoint found)
        """
        latest_path = self.find_latest_checkpoint()
        if latest_path is None:
            print("No checkpoint found, starting from scratch")
            return 0
        
        print(f"Loading checkpoint: {latest_path}")
        epoch = self.load_checkpoint(latest_path)
        print(f"Loaded checkpoint from epoch {epoch + 1}")
        next_epoch = epoch + 1
        
        if next_epoch >= self.num_epochs:
            print(f"Warning: Checkpoint is from epoch {epoch + 1}, but num_epochs={self.num_epochs}. "
                  f"Training will start from epoch {next_epoch}. Increase num_epochs to continue training.")
        
        return next_epoch
    
    def train(self, resume: bool = True) -> Dict[str, List[float]]:
        """Train the model. Automatically loads final_model.pt if it exists.
        
        Args:
            resume: If True, automatically load final_model.pt if it exists (default: True)
        """

        if self.test_loader is not None:
            logger.info(f"Test batches: {len(self.test_loader)}")
        logger.info("-" * 60)
        start_epoch = 0
        if resume:
            start_epoch = self.load_latest_checkpoint()
        
        logger.info(f"Training on {self.device}")
        logger.info(f"Total epochs: {self.num_epochs}")
        logger.info(f"Starting from epoch: {start_epoch + 1}")
        logger.info(f"Batch size: {self.train_loader.batch_size}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        if self.test_loader is not None:
            logger.info(f"Test batches: {len(self.test_loader)}")
        logger.info("-" * 60)
        
        for epoch in range(start_epoch, self.num_epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch()
            test_loss = self.validate()
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            self.history['learning_rates'].append(current_lr)
            
            epoch_time = time.time() - start_time
            
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Test Loss: {test_loss:.6f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.2f}s")
            
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            if self.early_stopping_patience != -1 and self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Save model at the end of training
        final_epoch = epoch
        self.save_checkpoint(final_epoch)
        print("-" * 60)
        logger.info(f"Training completed. Best Test Loss: {self.best_test_loss:.6f}")
        if self.checkpoint_dir is not None:
            logger.info(f"Final model saved to {self.checkpoint_dir / 'final_model.pt'}")
        
        if self.logger is not None:
            final_train_loss = self.history['train_loss'][-1]
            final_test_loss = self.history['test_loss'][-1] if self.history['test_loss'] else 0.0
            num_params = self.count_parameters(self.model)
            dataset_shape = self.get_dataset_shape()
            
            # CSV for now but we might want to use a proper db instead
            self.logger.log(
                run_type='training',
                dataset_name=self.dataset_name,
                dataset_shape=dataset_shape,
                dataset_size=self.train_dataset.df.memory_usage().sum(),
                model_parameters=num_params,
                context_window=self.train_dataset.context_window,
                prediction_length=self.train_dataset.prediction_length,
                epoch=epoch + 1,
                total_epochs=self.num_epochs,
                loss_type=self.loss_name,
                training_loss=final_train_loss,
                validation_loss=final_test_loss,
                training_history=self.history['train_loss'],
                validation_history=self.history['test_loss'],
            )
            print(f"Training run logged to {self.logger.csv_path}")
        
        return self.history
    
    def predict(self, context: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            context = context.to(self.device)
            predictions = self.model(context)
        return predictions.cpu()


class TanaForecastTester:
    """
    Tester class for evaluating trained models on test datasets.
    """
    def __init__(
        self,
        model: nn.Module,
        test_dataset: TimeSeriesDataset,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger: Optional[Logger] = None,
        dataset_name: str = "test"
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.test_dataset = test_dataset
        self.logger = logger
        self.dataset_name = dataset_name
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if device == "cuda" else False
        )
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count the total number of trainable parameters in a model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def get_dataset_shape(self) -> Tuple[int, int, int]:
        """
        Get the shape of the dataset.
        Returns: (num_samples, num_features, context_window)
        """
        num_samples = len(self.test_dataset)
        context, _ = self.test_dataset[0]
        num_features, context_window = context.shape
        return (num_samples, num_features, context_window)
    
    def evaluate(self, criterion: Optional[nn.Module] = None) -> Dict[str, float]:
        """
        Evaluate the model on the test dataset.
        
        Args:
            criterion: Loss function to use (default: MSELoss)
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if criterion is None:
            criterion = nn.MSELoss()
        
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        print(f"Evaluating on {self.device}")
        print(f"Test batches: {len(self.test_loader)}")
        print("-" * 60)
        
        with torch.no_grad():
            for context, target in self.test_loader:
                context = context.to(self.device)
                target = target.to(self.device)
                
                if target.dim() == 3 and target.size(1) == 1:
                    target = target.squeeze(1)
                
                predictions = self.model(context)
                
                # Compute losses
                mse_loss = criterion(predictions, target)
                mae_loss = torch.mean(torch.abs(predictions - target))
                
                total_loss += mse_loss.item()
                total_mae += mae_loss.item()
                num_batches += 1
                
                # Store for additional metrics
                all_predictions.append(predictions.cpu())
                all_targets.append(target.cpu())
        
        avg_mse = total_loss / num_batches
        avg_mae = total_mae / num_batches
        avg_rmse = np.sqrt(avg_mse)
        avg_mape = np.mean(np.abs((all_targets - all_predictions) / all_targets))
        avg_smape = np.mean(np.abs((all_targets - all_predictions) / (all_targets + all_predictions) / 2))

        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute R² score
        ss_res = torch.sum((all_targets - all_predictions) ** 2)
        ss_tot = torch.sum((all_targets - all_targets.mean()) ** 2)
        r2_score = 1 - (ss_res / ss_tot).item()

        
        metrics = {
            'mse': avg_mse,
            'mae': avg_mae,
            'rmse': avg_rmse,
            'r2': r2_score,
            'mape': avg_mape,
            'smape': avg_smape
        }
        
        print(f"Test MSE:  {avg_mse:.6f}")
        print(f"Test MAE:  {avg_mae:.6f}")
        print(f"Test RMSE: {avg_rmse:.6f}")
        print(f"Test R²:   {r2_score:.6f}")
        print(f"Test MAPE: {avg_mape:.6f}")
        print(f"Test SMAPE: {avg_smape:.6f}")
        print("-" * 60)
        
        # Log to CSV if logger is provided
        if self.logger is not None:
            num_params = self.count_parameters(self.model)
            dataset_shape = self.get_dataset_shape()
            loss_type = criterion.__class__.__name__
            
            self.logger.log_run(
                dataset_name=f"{self.dataset_name}_test",
                dataset_shape=dataset_shape,
                model_parameters=num_params,
                epoch=0,  # 0 for test runs
                total_epochs=0,
                loss_type=loss_type,
                training_loss=avg_mse,
                validation_loss=avg_mae,
                run_type='testing'
            )
            print(f"Test run logged to {self.logger.csv_path}")
        
        return metrics
    
    def predict(self, context: torch.Tensor) -> torch.Tensor:
        """
        Make predictions for a single context tensor.
        
        Args:
            context: Input context tensor
        
        Returns:
            Predictions tensor
        """
        self.model.eval()
        with torch.no_grad():
            context = context.to(self.device)
            predictions = self.model(context)
        return predictions.cpu()
    
    def predict_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions for all samples in the test dataset.
        
        Returns:
            Tuple of (predictions, targets)
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for context, target in self.test_loader:
                context = context.to(self.device)
                predictions = self.model(context)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(target.cpu())
        
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        return predictions, targets


if __name__ == "__main__":
    from src.model import TanaForecast
    
    # Create sample data
    df = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
    })
    
    # Create datasets
    train_dataset = TimeSeriesDataset(
        df=df[:700],
        context_window=100,
        prediction_length=12,
        stride=1,
        normalize=True
    )
    
    val_dataset = TimeSeriesDataset(
        df=df[600:850],
        context_window=100,
        prediction_length=12,
        stride=1,
        normalize=True
    )
    
    test_dataset = TimeSeriesDataset(
        df=df[800:],
        context_window=100,
        prediction_length=12,
        stride=1,
        normalize=True
    )
    
    # Initialize logger
    logger = Logger(csv_path='src/logs/training_logs.csv')
    
    # Create model
    model = TanaForecast(context_window=100, prediction_length=12)
    
    # Train
    trainer = TanaForecastTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=32,
        learning_rate=1e-3,
        num_epochs=5,
        checkpoint_dir='checkpoints/example',
        logger=logger,
        dataset_name='example_dataset'
    )
    
    history = trainer.train()
    
    # Test
    tester = TanaForecastTester(
        model=model,
        test_dataset=test_dataset,
        batch_size=32,
        logger=logger,
        dataset_name='example_dataset'
    )
    
    metrics = tester.evaluate()
    
    # Make a single prediction
    context, target = test_dataset[0]
    prediction = tester.predict(context.unsqueeze(0))
    print(f"\nSingle prediction shape: {prediction.shape}")
    print(f"Target shape: {target.shape}")
    
    # View logged runs
    print("\n" + "="*60)
    print("Logged Training Runs:")
    print("="*60)
    print(logger.read_logs())