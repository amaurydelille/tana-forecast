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

class Logger:
    """
    Logger class for recording training runs to CSV.
    Columns: dataset_name, dataset_shape, model_parameters, epochs, loss_type, training_loss, validation_loss
    """
    def __init__(self, csv_path: str) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create CSV with headers if it doesn't exist
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['run_type', 'dataset_name', 'dataset_shape', 'model_parameters', 
                               'epoch', 'total_epochs', 'loss_type', 'training_loss', 'validation_loss'])
    
    def log_run(
        self,
        dataset_name: str,
        dataset_shape: Tuple[int, ...],
        model_parameters: int,
        epoch: int,
        total_epochs: int,
        loss_type: str,
        training_loss: float,
        validation_loss: float,
        run_type: str = 'training'
    ) -> None:
        """
        Log a training run to the CSV file.
        
        Args:
            dataset_name: Name of the dataset used
            dataset_shape: Shape of the dataset (e.g., (samples, features, sequence_length))
            model_parameters: Total number of trainable parameters in the model
            epoch: Current epoch number
            total_epochs: Total number of epochs
            loss_type: Type of loss function used (e.g., 'MSE', 'MAE', 'TimeMoE')
            training_loss: Training loss
            validation_loss: Validation loss
            run_type: Type of run ('training', 'testing', 'final')
        """
        # Convert dataset_shape tuple to string representation
        shape_str = str(dataset_shape)
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                run_type,
                dataset_name,
                shape_str,
                model_parameters,
                epoch,
                total_epochs,
                loss_type,
                f"{training_loss:.6f}",
                f"{validation_loss:.6f}"
            ])
    
    def read_logs(self) -> pd.DataFrame:
        """Read all logs from the CSV file."""
        if not self.csv_path.exists():
            return pd.DataFrame(columns=['run_type', 'dataset_name', 'dataset_shape', 'model_parameters', 
                                        'epoch', 'total_epochs', 'loss_type', 'training_loss', 'validation_loss'])
        return pd.read_csv(self.csv_path)
    
    def get_best_run(self, metric: str = 'validation_loss') -> Optional[pd.Series]:
        """
        Get the best run based on a metric.
        
        Args:
            metric: Metric to optimize (default: 'validation_loss')
        
        Returns:
            Series containing the best run, or None if no logs exist
        """
        df = self.read_logs()
        if df.empty:
            return None
        
        if metric in ['training_loss', 'validation_loss']:
            return df.loc[df[metric].idxmin()]
        else:
            return df.loc[df[metric].idxmax()]

class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        context_window: int,
        prediction_length: int,
        target_columns: Optional[List[str]] = None,
        feature_columns: Optional[List[str]] = None,
        stride: int = 1,
        normalize: bool = True
    ) -> None:
        self.df = df.copy()
        self.context_window = context_window
        self.prediction_length = prediction_length
        self.stride = stride
        
        if feature_columns is None:
            self.feature_columns = list(df.columns)
        else:
            self.feature_columns = feature_columns
            
        if target_columns is None:
            self.target_columns = self.feature_columns
        else:
            self.target_columns = target_columns
        
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
        val_dataset: Optional[TimeSeriesDataset] = None,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        num_epochs: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: Optional[str] = None,
        early_stopping_patience: int = -1,
        logger: Optional[Logger] = None,
        dataset_name: str = "unknown",
        loss_fn: Optional[callable] = None,
        loss_name: str = "MSE",
        loss_kwargs: Optional[Dict] = None
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.logger = logger
        self.dataset_name = dataset_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Loss function configuration
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
            pin_memory=True if device == "cuda" else False
        )
        
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if device == "cuda" else False
            )
        
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
            'val_loss': [],
            'learning_rates': []
        }
        
        self.best_val_loss = float('inf')
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
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for context, target in self.val_loader:
                context = context.to(self.device)
                target = target.to(self.device)
                
                if target.dim() == 3 and target.size(1) == 1:
                    target = target.squeeze(1)
                
                if hasattr(self.model, 'return_router_info') and self.model.return_router_info:
                    predictions, router_probs, expert_indices = self.model(context)
                    # Call loss function with router info
                    loss = self.criterion(
                        y_true=target,
                        y_pred=predictions,
                        router_probs=router_probs,
                        expert_indices=expert_indices,
                        **self.loss_kwargs
                    )
                else:
                    predictions = self.model(context)
                    # Simple loss function (MSE, MAE, Huber, Quantile, etc.)
                    if self.use_custom_loss:
                        loss = self.criterion(target, predictions, **self.loss_kwargs)
                    else:
                        loss = self.criterion(predictions, target)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        if self.checkpoint_dir is None:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, path)
        
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
    
    def train(self) -> Dict[str, List[float]]:
        print(f"Training on {self.device}")
        print(f"Total epochs: {self.num_epochs}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val batches: {len(self.val_loader)}")
        print("-" * 60)
        
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(current_lr)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{self.num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.2f}s")
            
            # Log every epoch if logger is provided
            if self.logger is not None:
                num_params = self.count_parameters(self.model)
                dataset_shape = self.get_dataset_shape()
                
                self.logger.log_run(
                    dataset_name=self.dataset_name,
                    dataset_shape=dataset_shape,
                    model_parameters=num_params,
                    epoch=epoch + 1,
                    total_epochs=self.num_epochs,
                    loss_type=self.loss_name,
                    training_loss=train_loss,
                    validation_loss=val_loss,
                    run_type='training'
                )
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"  → New best model saved (Val Loss: {val_loss:.6f})")
            else:
                self.epochs_without_improvement += 1
            
            if self.early_stopping_patience != -1 and self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
        
        print("-" * 60)
        print(f"Training completed. Best Val Loss: {self.best_val_loss:.6f}")
        
        # Log the final summary if logger is provided
        if self.logger is not None:
            final_train_loss = self.history['train_loss'][-1]
            final_val_loss = self.history['val_loss'][-1] if self.history['val_loss'] else 0.0
            num_params = self.count_parameters(self.model)
            dataset_shape = self.get_dataset_shape()
            
            self.logger.log_run(
                dataset_name=self.dataset_name,
                dataset_shape=dataset_shape,
                model_parameters=num_params,
                epoch=epoch + 1,  # actual number of epochs trained
                total_epochs=self.num_epochs,
                loss_type=self.loss_name,
                training_loss=final_train_loss,
                validation_loss=final_val_loss,
                run_type='final'
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