import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import time
import csv
from src.utils import TimeStamps
import logging
import re
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Professional checkpoint management system with versioning and metadata.
    
    Features:
    - Unique checkpoint identification per dataset/run
    - Metadata tracking (hyperparameters, dataset info, timestamps)
    - Smart resume logic based on training state
    - Checkpoint versioning and cleanup
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        dataset_name: str,
        run_id: Optional[str] = None,
        keep_last_n: int = 3
    ):
        """
        Args:
            checkpoint_dir: Base directory for all checkpoints
            dataset_name: Name of the dataset being trained
            run_id: Optional unique run identifier (auto-generated if None)
            keep_last_n: Number of checkpoint versions to keep (-1 for all)
        """
        self.base_dir = Path(checkpoint_dir)
        self.dataset_name = self._sanitize_name(dataset_name)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.keep_last_n = keep_last_n
        
        self.run_dir = self.base_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_path = self.run_dir / "metadata.json"
        self.best_checkpoint_path = self.run_dir / "best_model.pt"
        self.last_checkpoint_path = self.run_dir / "last_model.pt"
        
        logger.info(f"CheckpointManager initialized for '{dataset_name}' (run_id: {self.run_id})")
        logger.info(f"Checkpoint directory: {self.run_dir}")
    
    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize names for filesystem compatibility."""
        sanitized = re.sub(r'[^A-Za-z0-9_.-]+', '_', name.strip())
        return sanitized if sanitized else "dataset"
    
    def save_checkpoint(
        self,
        epoch: int,
        model_state: Dict,
        optimizer_state: Dict,
        scheduler_state: Dict,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> None:
        """
        Save a checkpoint with full state and metadata.
        
        Args:
            epoch: Current epoch number
            model_state: Model state dict
            optimizer_state: Optimizer state dict
            scheduler_state: Scheduler state dict
            metrics: Dictionary of metric values (train_loss, val_loss, etc.)
            metadata: Additional metadata to store
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': scheduler_state,
            'metrics': metrics,
            'metadata': metadata or {},
            'timestamp': time.time(),
            'run_id': self.run_id,
            'dataset_name': self.dataset_name
        }
        
        torch.save(checkpoint, self.last_checkpoint_path)
        
        if is_best:
            torch.save(checkpoint, self.best_checkpoint_path)
            logger.info(f"New best checkpoint saved (epoch {epoch + 1})")
        
        self._update_metadata(epoch, metrics, metadata)
        
        if self.keep_last_n > 0:
            self._cleanup_old_checkpoints()
    
    def load_checkpoint(
        self,
        checkpoint_type: str = "last"
    ) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_type: "last", "best", or path to specific checkpoint
            
        Returns:
            Checkpoint dictionary or None if not found
        """
        if checkpoint_type == "last":
            checkpoint_path = self.last_checkpoint_path
        elif checkpoint_type == "best":
            checkpoint_path = self.best_checkpoint_path
        else:
            checkpoint_path = Path(checkpoint_type)
        
        if not checkpoint_path.exists():
            logger.info(f"No checkpoint found at {checkpoint_path}")
            return None
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"Checkpoint loaded: epoch {checkpoint['epoch'] + 1}, "
                       f"metrics: {checkpoint.get('metrics', {})}")
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            logger.warning(f"Deleting corrupted checkpoint: {checkpoint_path}")
            try:
                checkpoint_path.unlink()
                if self.metadata_path.exists():
                    logger.warning(f"Deleting metadata file: {self.metadata_path}")
                    self.metadata_path.unlink()
            except Exception as del_e:
                logger.error(f"Failed to delete corrupted checkpoint: {del_e}")
            return None
    
    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoints for this run."""
        checkpoints = sorted(self.run_dir.glob("checkpoint_epoch_*.pt"))
        return checkpoints
    
    def _update_metadata(
        self,
        epoch: int,
        metrics: Dict[str, float],
        additional_metadata: Optional[Dict] = None
    ) -> None:
        """Update or create metadata file."""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {
                'dataset_name': self.dataset_name,
                'run_id': self.run_id,
                'created_at': datetime.now().isoformat(),
                'epochs': []
            }
        
        epoch_info = {
            'epoch': epoch + 1,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_metadata:
            epoch_info.update(additional_metadata)
        
        metadata['epochs'].append(epoch_info)
        metadata['last_updated'] = datetime.now().isoformat()
        metadata['total_epochs'] = epoch + 1
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) > self.keep_last_n:
            to_remove = checkpoints[:-self.keep_last_n]
            for ckpt_path in to_remove:
                ckpt_path.unlink()
                logger.debug(f"Removed old checkpoint: {ckpt_path.name}")
    
    def get_latest_run_id(
        base_dir: Path,
        dataset_name: str
    ) -> Optional[str]:
        """Get the latest run_id for a dataset (for resuming)."""
        if not base_dir.exists():
            return None
        
        metadata_file = base_dir / "metadata.json"
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                return metadata.get('run_id')
        
        return None


class TrainingLogger:
    def __init__(
        self,
        log_dir: Path,
        dataset_name: str,
        run_id: str,
        context_window: int,
        prediction_length: int,
        batch_size: int,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.context_window = context_window
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        
        sanitized_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', dataset_name)
        self.log_file = self.log_dir / f"{sanitized_name}_{run_id}.csv"
        
        self.columns = [
            'run_id',
            'dataset_name',
            'context_window',
            'prediction_length',
            'batch_size',
            'epoch',
            'train_loss',
            'val_loss',
            'learning_rate',
            'epoch_time_sec',
            'total_time_sec',
            'best_val_loss',
            'timestamp',
            'date'
        ]
        
        self._initialize_log_file()
        self.start_time = time.time()
        
        logger.info(f"TrainingLogger initialized: {self.log_file}")
    
    def _initialize_log_file(self) -> None:
        """Create log file with headers if it doesn't exist."""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                writer.writeheader()
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
        epoch_time: float,
        best_val_loss: float
    ) -> None:
        """
        Log a single epoch's metrics.
        
        Args:
            epoch: Current epoch number (0-indexed)
            train_loss: Training loss
            val_loss: Validation loss
            learning_rate: Current learning rate
            epoch_time: Time taken for this epoch (seconds)
            best_val_loss: Best validation loss so far
        """
        total_time = time.time() - self.start_time
        timestamp = time.time()
        date = datetime.fromtimestamp(timestamp).isoformat()
        
        row = {
            'run_id': self.run_id,
            'dataset_name': self.dataset_name,
            'context_window': self.context_window,
            'prediction_length': self.prediction_length,
            'batch_size': self.batch_size,
            'epoch': epoch + 1,
            'train_loss': f"{train_loss:.6f}",
            'val_loss': f"{val_loss:.6f}",
            'learning_rate': f"{learning_rate:.2e}",
            'epoch_time_sec': f"{epoch_time:.2f}",
            'total_time_sec': f"{total_time:.2f}",
            'best_val_loss': f"{best_val_loss:.6f}",
            'timestamp': timestamp,
            'date': date
        }
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(row)
    
    def log_summary(self, summary_data: Dict[str, Any]) -> None:
        """
        Log a summary of the entire training run.
        
        Args:
            summary_data: Dictionary containing summary statistics
        """
        summary_file = self.log_dir / f"{self.dataset_name}_{self.run_id}_summary.json"
        
        summary_data['timestamp'] = datetime.now().isoformat()
        summary_data['run_id'] = self.run_id
        summary_data['dataset_name'] = self.dataset_name
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Training summary saved: {summary_file}")

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
        
        numeric_feature_cols = self.df[self.feature_columns].select_dtypes(include=["number", "bool"]).columns.tolist()
        if len(numeric_feature_cols) != len(self.feature_columns):
            import logging
            logging.getLogger(__name__).warning(
                f"Dropping non-numeric feature columns: "
                f"{set(self.feature_columns) - set(numeric_feature_cols)}"
            )
        self.feature_columns = numeric_feature_cols
            
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
    """
    Professional trainer for TanaForecast models with modern checkpoint and logging.
    
    Features:
    - Automatic checkpoint management with versioning
    - Structured per-epoch logging
    - Smart resume logic
    - Best model tracking
    """
    
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
        log_dir: Optional[str] = None,
        early_stopping_patience: int = -1,
        dataset_name: str = "unknown",
        run_id: Optional[str] = None,
        loss_fn: Optional[callable] = None,
        loss_name: str = "MSE",
        loss_kwargs: Optional[Dict] = None,
        keep_checkpoints: int = 3,
        resume_from_checkpoint: bool = True
    ) -> None:
        """
        Args:
            model: Model to train
            train_dataset: Training dataset
            test_dataset: Optional validation/test dataset
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            num_epochs: Total number of epochs to train
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save training logs
            early_stopping_patience: Patience for early stopping (-1 to disable)
            dataset_name: Name of the dataset
            run_id: Optional unique run identifier (auto-generated if None)
            loss_fn: Custom loss function
            loss_name: Name of the loss function for logging
            loss_kwargs: Additional kwargs for loss function
            keep_checkpoints: Number of checkpoint versions to keep
            resume_from_checkpoint: Whether to resume from latest checkpoint
        """
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.resume_from_checkpoint = resume_from_checkpoint
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        

        self.loss_name = loss_name
        self.loss_kwargs = loss_kwargs or {}
        if loss_fn is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = loss_fn
        self.use_custom_loss = loss_fn is not None
        
        pin_memory = self.device == "cuda" or (isinstance(self.device, torch.device) and self.device.type == "cuda")
        num_workers = 2 if pin_memory else 0
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
        
        if test_dataset is not None:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=num_workers > 0
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
        
        if checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=Path(checkpoint_dir),
                dataset_name=dataset_name,
                run_id=run_id,
                keep_last_n=keep_checkpoints
            )
            self.run_id = self.checkpoint_manager.run_id
        else:
            self.checkpoint_manager = None
            self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if log_dir:
            self.training_logger = TrainingLogger(
                log_dir=Path(log_dir),
                dataset_name=dataset_name,
                run_id=self.run_id,
                context_window=self.train_dataset.context_window,
                prediction_length=self.train_dataset.prediction_length,
                batch_size=self.batch_size,
            )
        else:
            self.training_logger = None
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.start_epoch = 0
        
    
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
        
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

        for context, target in self.train_loader:
            context = context.to(self.device)
            target = target.to(self.device)
            
            if target.dim() == 3 and target.size(1) == 1:
                target = target.squeeze(1)
            
            self.optimizer.zero_grad()
            
            if self.device == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
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
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
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
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
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
                
                if self.device == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
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
                else:
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
    
    def _load_checkpoint_state(self) -> None:
        """Load checkpoint state if resume is enabled."""
        if not self.resume_from_checkpoint or self.checkpoint_manager is None:
            return
        
        checkpoint = self.checkpoint_manager.load_checkpoint("last")
        if checkpoint is None:
            return
        
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            checkpoint_epoch = checkpoint['epoch'] + 1
            if checkpoint_epoch >= self.num_epochs:
                logger.info(f"Checkpoint at epoch {checkpoint_epoch} is beyond num_epochs ({self.num_epochs}). Starting fresh training.")
                self.start_epoch = 0
                self.best_val_loss = float('inf')
                self.history = {
                    'train_loss': [],
                    'val_loss': [],
                    'learning_rates': []
                }
            else:
                self.start_epoch = checkpoint_epoch
                self.history = checkpoint.get('metrics', {}).get('history', self.history)
                self.best_val_loss = checkpoint.get('metrics', {}).get('best_val_loss', float('inf'))
                logger.info(f"Resumed from epoch {self.start_epoch}, best_val_loss: {self.best_val_loss:.6f}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint state: {e}")
            logger.warning("Deleting incompatible checkpoint and starting fresh")
            try:
                last_checkpoint = self.checkpoint_manager.run_dir / "last_model.pt"
                if last_checkpoint.exists():
                    last_checkpoint.unlink()
                best_checkpoint = self.checkpoint_manager.run_dir / "best_model.pt"
                if best_checkpoint.exists():
                    best_checkpoint.unlink()
                metadata_path = self.checkpoint_manager.run_dir / "metadata.json"
                if metadata_path.exists():
                    metadata_path.unlink()
            except Exception as del_e:
                logger.error(f"Failed to delete incompatible checkpoint: {del_e}")
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model with professional checkpoint and logging management.
        
        Returns:
            Dictionary containing training history
        """
        self._load_checkpoint_state()
        
        logger.info("=" * 70)
        logger.info(f"Training: {self.dataset_name} (run_id: {self.run_id})")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.start_epoch + 1} -> {self.num_epochs}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        if self.test_loader:
            logger.info(f"Val batches: {len(self.test_loader)}")
        logger.info(f"Model parameters: {self.count_parameters(self.model):,}")
        logger.info(f"Loss: {self.loss_name}")
        logger.info("=" * 70)
        
        training_start_time = time.time()
        
        for epoch in range(self.start_epoch, self.num_epochs):
            epoch_start_time = time.time()
            
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(current_lr)
            
            epoch_time = time.time() - epoch_start_time
            
            if self.test_loader is not None:
                metric = val_loss
            else:
                metric = train_loss

            is_best = metric < self.best_val_loss
            if is_best:
                self.best_val_loss = metric
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} | "
                f"Train: {train_loss:.6f} | "
                f"Val: {val_loss:.6f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.2f}s"
                f"{' [BEST]' if is_best else ''}"
            )
            
            if self.checkpoint_manager:
                metrics = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': current_lr,
                    'best_val_loss': self.best_val_loss,
                    'history': self.history
                }
                metadata = {
                    'batch_size': self.batch_size,
                    'learning_rate': self.learning_rate,
                    'loss_name': self.loss_name
                }
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    model_state=self.model.state_dict(),
                    optimizer_state=self.optimizer.state_dict(),
                    scheduler_state=self.scheduler.state_dict(),
                    metrics=metrics,
                    metadata=metadata,
                    is_best=is_best
                )
            
            if self.training_logger:
                self.training_logger.log_epoch(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=current_lr,
                    epoch_time=epoch_time,
                    best_val_loss=self.best_val_loss
                )
            
            if self.early_stopping_patience > 0 and self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        total_training_time = time.time() - training_start_time
        
        logger.info("=" * 70)
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        logger.info(f"Total training time: {total_training_time:.2f}s")
        logger.info("=" * 70)
        
        if self.training_logger:
            summary = {
                'total_epochs': len(self.history['train_loss']),
                'best_val_loss': self.best_val_loss,
                'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else 0,
                'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else 0,
                'total_time_sec': total_training_time,
                'model_parameters': self.count_parameters(self.model),
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'loss_name': self.loss_name
            }
            self.training_logger.log_summary(summary)
        
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
        training_logger: Optional[TrainingLogger] = None,
        dataset_name: str = "test"
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.test_dataset = test_dataset
        self.logger = training_logger
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
                
                mse_loss = criterion(predictions, target)
                mae_loss = torch.mean(torch.abs(predictions - target))
                
                total_loss += mse_loss.item()
                total_mae += mae_loss.item()
                num_batches += 1
                
                all_predictions.append(predictions.cpu())
                all_targets.append(target.cpu())
        
        avg_mse = total_loss / num_batches
        avg_mae = total_mae / num_batches
        avg_rmse = np.sqrt(avg_mse)
        avg_mape = np.mean(np.abs((all_targets - all_predictions) / all_targets))
        avg_smape = np.mean(np.abs((all_targets - all_predictions) / (all_targets + all_predictions) / 2))

        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
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
        
        if self.logger is not None:
            num_params = self.count_parameters(self.model)
            dataset_shape = self.get_dataset_shape()
            loss_type = criterion.__class__.__name__
            
            self.logger.log_run(
                dataset_name=f"{self.dataset_name}_test",
                dataset_shape=dataset_shape,
                model_parameters=num_params,
                epoch=0,
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