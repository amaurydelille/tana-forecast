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
        early_stopping_patience: int = -1
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        
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
        
        self.criterion = nn.MSELoss()
        
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
            
            predictions = self.model(context)
            
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
                
                predictions = self.model(context)
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
        
        return self.history
    
    def predict(self, context: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            context = context.to(self.device)
            predictions = self.model(context)
        return predictions.cpu()


if __name__ == "__main__":
    from src.model import Decoder
    
    df = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
    })
    
    train_dataset = TimeSeriesDataset(
        df=df[:800],
        context_window=100,
        prediction_length=12,
        stride=1,
        normalize=True
    )
    
    val_dataset = TimeSeriesDataset(
        df=df[700:],
        context_window=100,
        prediction_length=12,
        stride=1,
        normalize=True
    )
    
    model = Decoder(context_window=100, prediction_length=12, d_model=2)
    
    trainer = TanaForecastTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=32,
        learning_rate=1e-3,
        num_epochs=5,
        checkpoint_dir='checkpoints'
    )
    
    history = trainer.train()
    
    context, target = val_dataset[0]
    prediction = trainer.predict(context.unsqueeze(0))
    print(f"\nPrediction shape: {prediction.shape}")
    print(f"Target shape: {target.shape}")