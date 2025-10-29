import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, Callable

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
        from src.model import N_EXPERT, TOP_K
        n_experts = kwargs.get('n_experts', N_EXPERT)
        top_k = kwargs.get('top_k', TOP_K)
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
            router_probs: [batch_size, seq_len, n_experts] - softmax probabilities
            expert_indices: [batch_size, seq_len, top_k] - selected expert indices
            n_experts: int - total number of experts (N)
            top_k: int - number of experts selected per step (K)
        
        Returns:
            aux_loss: scalar tensor
        """
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
