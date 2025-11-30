import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Callable, Union, Iterable
import pandas as pd
import numpy as np
import datetime

def get_loss_config(loss_type: str, **kwargs) -> Tuple[Optional[Callable], str, Dict, bool]:
    """
    Factory function to get loss configuration.
    
    Args:
        loss_type: One of 'mse', 'mae', 'huber', 'quantile', 'timemoe'
        **kwargs: Additional parameters for the loss function
            - For huber: delta (default: 1.0)
            - For quantile: q (default: 0.5)
            - For timemoe: n_experts, top_k, alpha (default: 0.01)
    
    Returns:
        Tuple of (loss_fn, loss_name, loss_kwargs, return_router_info)
    
    Example:
        >>> loss_fn, name, kwargs, router_info = get_loss_config('mae')
        >>> trainer = TanaForecastTrainer(
        ...     model=model,
        ...     loss_fn=loss_fn,
        ...     loss_name=name,
        ...     loss_kwargs=kwargs
        ... )
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'mse':
        return None, 'MSE', {}, False
    
    elif loss_type == 'mae':
        return Loss.mae_loss, 'MAE', {}, False
    
    elif loss_type == 'huber':
        delta = kwargs.get('delta', 1.0)
        return Loss.huber_loss, 'Huber', {'delta': delta}, False
    
    elif loss_type == 'quantile':
        q = kwargs.get('q', 0.5)
        return Loss.quantile_loss, f'Quantile_{q}', {'q': q}, False
    
    elif loss_type == 'timemoe':
        n_experts = kwargs.get('n_experts', Constants.n_experts)
        top_k = kwargs.get('top_k', Constants.top_k)
        alpha = kwargs.get('alpha', 0.01)
        return (
            Loss.time_moe_loss,
            'TimeMoE',
            {'n_experts': n_experts, 'top_k': top_k, 'alpha': alpha},
            True  # MUST return router info
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from: mse, mae, huber, quantile, timemoe")

class Loss:
    @staticmethod
    def quantile_loss(z: torch.Tensor, z_q: torch.Tensor, q: float) -> torch.Tensor:
        """
        Implements the quantile loss described as in AWS/Chronos-2: https://www.arxiv.org/pdf/2510.15821
        
        Args:
            z: true values [...]
            z_q: predicted quantile values [...]
            q: quantile level (e.g., 0.5 for median, 0.9 for 90th percentile)
        
        Returns:
            quantile loss tensor
        """
        errors = z - z_q
        return torch.mean(torch.max(q * errors, (q - 1) * errors))

    @staticmethod
    def huber_loss(x: torch.Tensor, x_hat: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        """
        Implements the Huber loss (robust to outliers).
        
        Args:
            x: true values [...]
            x_hat: predicted values [...]
            delta: threshold at which to switch from quadratic to linear
        
        Returns:
            huber loss tensor
        """
        error = torch.abs(x - x_hat)
        quadratic = torch.clamp(error, max=delta)
        linear = error - quadratic
        return torch.mean(0.5 * quadratic ** 2 + delta * linear)

    @staticmethod
    def auxiliary_loss(router_probs: torch.Tensor, expert_indices: torch.Tensor, n_experts: int, top_k: int) -> torch.Tensor:
        """
        Implements the auxiliary load balancing loss from TimeMoE/Switch Transformer.
        
        L_aux = N × Σ(f_i × r_i)
        where:
        - f_i is the fraction of tokens assigned to expert i
        - r_i is the average router probability for expert i
        
        Args:
            router_probs: [batch_size, seq_len, n_experts] (or with n_decoders dim)
            expert_indices: [batch_size, seq_len, top_k] (or with n_decoders dim)
            n_experts: int - total number of experts (N)
            top_k: int - number of experts selected per step (K)
        
        Returns:
            aux_loss: scalar tensor
        """
        if router_probs.dim() == 4:
            # Shape: [n_decoders, batch_size, seq_len, n_experts]
            # Collapse decoders into batch dimension
            router_probs = router_probs.view(-1, router_probs.size(2), router_probs.size(3))
            expert_indices = expert_indices.view(-1, expert_indices.size(2), expert_indices.size(3))

        batch_size, T, _ = router_probs.shape
        
        # r_i: average router probability for each expert across all positions
        # Shape: [n_experts]
        r = router_probs.mean(dim=[0, 1])
        
        # f_i: fraction of times each expert was selected
        total_selections = batch_size * T * top_k
        
        # Count selections per expert
        expert_counts = torch.zeros(n_experts, device=router_probs.device)
        for i in range(n_experts):
            expert_counts[i] = (expert_indices == i).sum().float()
        
        f = expert_counts / total_selections
        
        # Compute auxiliary loss
        aux_loss = n_experts * (f * r).sum()
        
        return aux_loss

    @staticmethod
    def time_moe_loss(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        router_probs: torch.Tensor, 
        expert_indices: torch.Tensor, 
        n_experts: int, 
        top_k: int,
        alpha: float = 0.01
    ) -> torch.Tensor:
        """
        Implements the complete TimeMoE loss: MSE + auxiliary load balancing loss.
        
        L_total = MSE(y_true, y_pred) + α × L_aux
        
        Args:
            y_true: ground truth values [batch_size, ...]
            y_pred: predicted values [batch_size, ...]
            router_probs: [batch_size, seq_len, n_experts] - softmax probabilities
            expert_indices: [batch_size, seq_len, top_k] - selected expert indices
            n_experts: int - total number of experts (N)
            top_k: int - number of experts selected per step (K)
            alpha: weight for auxiliary loss (typically small, e.g., 0.01)
        
        Returns:
            total loss: scalar tensor
        """
        # Main prediction loss (MSE)
        mse_loss = F.mse_loss(y_pred, y_true)
        
        # Auxiliary load balancing loss
        aux_loss = Loss.auxiliary_loss(router_probs, expert_indices, n_experts, top_k)
        
        # Total loss
        total_loss = mse_loss + alpha * aux_loss
        
        return total_loss
    
    @staticmethod
    def mse_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Mean Squared Error loss (wrapper for convenience).
        
        Args:
            y_true: ground truth values
            y_pred: predicted values
        
        Returns:
            MSE loss
        """
        return F.mse_loss(y_pred, y_true)
    
    @staticmethod
    def mae_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Mean Absolute Error loss.
        
        Args:
            y_true: ground truth values
            y_pred: predicted values
        
        Returns:
            MAE loss
        """
        return F.l1_loss(y_pred, y_true)

class TimeStamps:
    @staticmethod
    def universal_timestamp_normalized(
        timestamps: Union[pd.Series, torch.Tensor, np.ndarray, Iterable[Union[datetime.datetime, pd.Timestamp, str, int, float]]],
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Universally normalize timestamps to days since epoch (1970-01-01).
        
        Supports multiple input formats:
        - pandas Series with datetime/timestamp dtypes
        - torch.Tensor or numpy arrays (treated as Unix timestamps in seconds)
        - Lists of datetime objects, pd.Timestamp, or strings
        - Numeric values (treated as Unix timestamps)
        
        Args:
            timestamps: Input timestamps in various formats
            device: device to create tensor on (default: "cpu")
        
        Returns:
            torch.Tensor: Normalized timestamps as float32 tensor (days since 1970-01-01)
        """
        SCALING_CONSTANT = 86400.0
        
        if isinstance(timestamps, torch.Tensor):
            timestamps_np = timestamps.cpu().numpy() if timestamps.is_cuda else timestamps.numpy()
            days_since_epoch = timestamps_np.astype(np.float64) / SCALING_CONSTANT
            return torch.tensor(days_since_epoch, dtype=torch.float32, device=device)
        
        if isinstance(timestamps, np.ndarray):
            if timestamps.dtype.kind in ['M', 'm']:
                timestamps = pd.Series(timestamps)
            else:
                days_since_epoch = timestamps.astype(np.float64) / SCALING_CONSTANT
                return torch.tensor(days_since_epoch, dtype=torch.float32, device=device)
        
        if isinstance(timestamps, pd.Series):
            if pd.api.types.is_datetime64_any_dtype(timestamps):
                min_timestamp = pd.Timestamp('1970-01-01')
                days_since_epoch = (timestamps - min_timestamp).dt.total_seconds() / SCALING_CONSTANT
                return torch.tensor(days_since_epoch.values, dtype=torch.float32, device=device)
            else:
                timestamps = timestamps.values
        
        min_timestamp = pd.Timestamp('1970-01-01')
        normalized = []
        
        for timestamp in timestamps:
            if isinstance(timestamp, (int, float)):
                days_since_epoch = float(timestamp) / SCALING_CONSTANT
                normalized.append(days_since_epoch)
            elif isinstance(timestamp, str):
                try:
                    ts = pd.Timestamp(timestamp)
                    days_since_epoch = (ts - min_timestamp).total_seconds() / SCALING_CONSTANT
                    normalized.append(float(days_since_epoch))
                except (ValueError, TypeError):
                    normalized.append(0.0)
            elif isinstance(timestamp, (datetime.datetime, pd.Timestamp)):
                days_since_epoch = (pd.Timestamp(timestamp) - min_timestamp).total_seconds() / SCALING_CONSTANT
                normalized.append(float(days_since_epoch))
            else:
                try:
                    ts = pd.Timestamp(timestamp)
                    days_since_epoch = (ts - min_timestamp).total_seconds() / SCALING_CONSTANT
                    normalized.append(float(days_since_epoch))
                except (ValueError, TypeError):
                    normalized.append(0.0)
        
        return torch.tensor(normalized, dtype=torch.float32, device=device)

    @staticmethod
    def infer_frequency(timestamps: Union[pd.Series, torch.Tensor, np.ndarray, Iterable[Union[datetime.datetime, pd.Timestamp, str, int, float]]]) -> str:
        """
        Infer the frequency of the timestamps.
        
        Args:
            timestamps: Input timestamps in various formats
        """
        if isinstance(timestamps, pd.Series):
            return timestamps.infer_freq()
        elif isinstance(timestamps, torch.Tensor) or isinstance(timestamps, np.ndarray):
            return pd.infer_freq(timestamps)
        else:
            raise ValueError(f"Unsupported type: {type(timestamps)}")

class Constants:
    @staticmethod
    def infer_context_window(frequency: str) -> int:
        """
        Infer the context window based on the frequency.
        For now we use roughly 2 years of data as the context window.
        Args:
            frequency: Frequency of the timestamps
        """
        raise NotImplementedError("Not implemented")
        if frequency == 'D':
            return 730
        elif frequency == 'W':
            return 104
        elif frequency == 'M':
            return 24
        elif frequency == 'Y':
            return 2
        elif frequency == 'H':
            pass

    context_window: int = 4096
    prediction_length: int = 256
    batch_size: int = 2048
    learning_rate: float = 1e-3
    num_epochs: int = 100
    patch_size: int = 24
    stride: int = 12
    d_out: int = 512
    n_experts: int = 8
    top_k: int = 4
    n_decoders: int = 12
    d_ff: int = 2048
    attention_heads: int = 16