import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List

PATCH_SIZE = 24
STRIDE = 12
D_OUT = 64
N_EXPERT = 8
TOP_K = 4
N_DECODERS = 6
D_FF = 2048

class Monitor:
    def __init__(self) -> None:
        self.expert_counts = [0] * N_EXPERT

    def update_expert_counts(self, expert_ids: torch.Tensor) -> None:
        for expert_id in expert_ids.unique():
            self.expert_counts[expert_id.item()] += 1

    def get_expert_counts(self) -> List[int]:
        return self.expert_counts

class Patch(nn.Module):
    def __init__(
        self, 
        X: torch.Tensor, 
        context_window: int, 
        patch_size: int, 
        stride: int,
        d_out: int
    ) -> None:
        """
        Args:
            X: torch.Tensor, the input tensor of shape (C, L)
            context_window: int, the context window size
            patch_size: int, the patch size
            stride: int, the stride size
            d_out: int, the embedding output dimension
        """
        super().__init__()
    
        assert stride < patch_size, "Stride must be strictly smaller than patch size."
        assert patch_size < context_window, "Patch size must be strictly smaller than the context window."
        
        C = X.shape[1]
        self.L = context_window
        self.P = patch_size
        self.S = stride
        self.d_out = d_out
        self.conv = nn.Conv1d(in_channels=C, out_channels=d_out, kernel_size=self.P, stride=self.S, padding=0, bias=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        We consider that dim(X) = (B, C, L), with B the batch size, C the number of channels (features)
        and L the context window.
        """
        X = self.conv(X).transpose(1, 2)
        _, N, D = X.shape
    
        position = torch.arange(N, device=X.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, device=X.device) * (-torch.log(torch.tensor(10000.0)) / D))
        
        pe = torch.zeros(N, D, device=X.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return X + pe

class CrossLayer(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.randn(d_model, d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        xl_w = torch.einsum('...d,de->...e', xl, self.weight)
        xl_w_x0 = xl_w * x0
        return xl_w_x0 + self.bias + xl

class DeepCrossNetwork(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        n_cross_layers: int = 3,
        n_deep_layers: int = 3,
        deep_hidden_dim: int = None,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_cross_layers = n_cross_layers
        self.n_deep_layers = n_deep_layers
        self.deep_hidden_dim = deep_hidden_dim or d_model * 2
        
        self.cross_layers = nn.ModuleList([
            CrossLayer(d_model) for _ in range(n_cross_layers)
        ])
        
        deep_layers = []
        in_dim = d_model
        for _ in range(n_deep_layers):
            deep_layers.extend([
                nn.Linear(in_dim, self.deep_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = self.deep_hidden_dim
        deep_layers.append(nn.Linear(self.deep_hidden_dim, d_model))
        self.deep_network = nn.Sequential(*deep_layers)
        
        self.fusion = nn.Linear(d_model * 2, d_model)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x0 = X
        xl = X
        
        for cross_layer in self.cross_layers:
            xl = cross_layer(x0, xl)
        
        xd = self.deep_network(X)
        
        combined = torch.cat([xl, xd], dim=-1)
        output = self.fusion(combined)
        
        return output

class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8) -> None:
        super().__init__()
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.q = nn.Linear(in_features=d_model, out_features=d_model)
        self.k = nn.Linear(in_features=d_model, out_features=d_model)
        self.v = nn.Linear(in_features=d_model, out_features=d_model)
        self.out = nn.Linear(in_features=d_model, out_features=d_model)
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: torch.Tensor, the input tensor of shape (B, N, D)
        """
        B, N, D = X.shape

        q = self.q(X).view(B, N, self.num_heads, self.d_model_per_head).transpose(1, 2)
        k = self.k(X).view(B, N, self.num_heads, self.d_model_per_head).transpose(1, 2)
        v = self.v(X).view(B, N, self.num_heads, self.d_model_per_head).transpose(1, 2)

        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_model_per_head)

        mask = torch.tril(torch.ones(N, N, device=X.device)) # this function stores the mask on CPU by default.
        mask = mask.masked_fill(mask == 0, float('-inf'))

        attention_scores = attention_scores + mask
        attention_weights = F.softmax(attention_scores, dim=-1)

        attended = attention_weights @ v
        attended = attended.transpose(1, 2).contiguous().view(B, N, D)
        return self.out(attended)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=d_ff, out_features=d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.fc1(X)))

class MixtureOfExpert(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        n_experts: int, 
        top_k: int,
        bias: bool = False,
        dropout: float = 0.2,
        monitor: Monitor = None,
        return_router_info: bool = False
    ) -> None:
        super().__init__()
        assert n_experts > top_k, "Number of experts must be strictly greater than top_k."
        assert 0 <= dropout <= 1, "Dropout must be between 0 and 1."

        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([FeedForward(d_model=d_model, d_ff=D_FF) for _ in range(n_experts)])
        self.router = nn.Linear(d_model, n_experts)
        self.out = nn.Linear(d_model, d_model)
        self.monitor = monitor
        self.return_router_info = return_router_info
        
    def forward(self, X: torch.Tensor):
        router_logits = self.router(X)
        router_weights = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = router_weights.topk(k=self.top_k, dim=-1)
        output = torch.zeros_like(X)
        
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, :, i]
            expert_weight = top_k_weights[:, :, i].unsqueeze(-1)
            self.monitor.update_expert_counts(expert_idx) if self.monitor is not None else None

            for expert_id in range(self.n_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_output = self.experts[expert_id](X)
                    mask_expanded = mask.unsqueeze(-1)
                    output += expert_weight * expert_output * mask_expanded.float()

        output = self.out(output)
        
        if self.return_router_info:
            return output, router_weights, top_k_indices
        return output

class Decoder(nn.Module):
    def __init__(self, 
        context_window: int,
        prediction_length: int,
        d_model: int = 64,
        monitor: Monitor = None,
        return_router_info: bool = False
    ) -> None:
        super().__init__()
        self.context_window = context_window
        self.prediction_length = prediction_length
        self.d_model = d_model
        self.monitor = monitor
        self.return_router_info = return_router_info
        
        self.patch = Patch(
            X=torch.zeros(1, d_model, context_window),
            context_window=context_window,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            d_out=D_OUT
        )
        self.deep_cross = DeepCrossNetwork(d_model=D_OUT, n_cross_layers=3, n_deep_layers=3, deep_hidden_dim=D_OUT * 2, dropout=0.1)
        self.layer_norm_1 = nn.LayerNorm(D_OUT)
        self.attention = MultiHeadCausalSelfAttention(d_model=D_OUT)
        self.layer_norm_2 = nn.LayerNorm(D_OUT)
        self.moe = MixtureOfExpert(d_model=D_OUT, n_experts=N_EXPERT, top_k=TOP_K, monitor=self.monitor, return_router_info=return_router_info)
        self.out = nn.Linear(in_features=D_OUT, out_features=prediction_length)

    def project_channels(self, X: torch.Tensor) -> torch.Tensor:
        B, C, L = X.shape
        
        if C == self.d_model:
            return X
        elif C < self.d_model:
            padding = torch.zeros(B, self.d_model - C, L, device=X.device)
            return torch.cat([X, padding], dim=1)
        else:
            pooled = F.adaptive_avg_pool1d(X.transpose(1, 2), self.d_model).transpose(1, 2)
            return pooled

    def forward(self, X: torch.Tensor, prediction_length: int = 12):
        X = self.project_channels(X)
        
        X = self.patch(X)
        X = self.deep_cross(X)
        X = self.layer_norm_1(X)
        X = self.attention(X)
        
        if self.return_router_info:
            X, router_probs, expert_indices = self.moe(X)
        else:
            X = self.moe(X)
            
        X = self.layer_norm_2(X)
        X = X[:, -1, :]
        output = self.out(X)
        
        if self.return_router_info:
            return output, router_probs, expert_indices
        return output

class TanaForecast(nn.Module):
    def __init__(
        self, 
        context_window: int, 
        prediction_length: int,
        return_router_info: bool = False,
        device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    ) -> None:
        super().__init__()

        self.device = device if device is not None else "mps" if torch.backends.mps.is_available() else "cpu"

        # to monitor router activation
        self.monitor = Monitor()

        self.context_window = context_window
        self.prediction_length = prediction_length
        self.return_router_info = return_router_info

        self.decoders = nn.ModuleList([
            Decoder(
                context_window=context_window, 
                prediction_length=prediction_length,
                monitor=self.monitor,
                return_router_info=return_router_info
            ) for _ in range(N_DECODERS)
        ])

        self.number_of_parameters = sum(p.numel() for p in self.parameters())

    def forward(self, X: torch.Tensor, prediction_length: int = 12):
        outputs = []
        router_probs_list = []
        expert_indices_list = []
        
        for decoder in self.decoders:
            if self.return_router_info:
                output, router_probs, expert_indices = decoder(X, prediction_length)
                outputs.append(output)
                router_probs_list.append(router_probs)
                expert_indices_list.append(expert_indices)
            else:
                output = decoder(X, prediction_length)
                outputs.append(output)
        
        stacked_outputs = torch.stack(outputs, dim=0)
        mean_output = torch.mean(stacked_outputs, dim=0)
        
        if self.return_router_info:
            stacked_router_probs = torch.stack(router_probs_list, dim=0)
            stacked_expert_indices = torch.stack(expert_indices_list, dim=0)
            return mean_output, stacked_router_probs, stacked_expert_indices
        
        return mean_output
    
    def set_return_router_info(self, value: bool) -> None:
        """
        Dynamically enable/disable router info return.
        Useful for switching between different loss functions during training.
        """
        self.return_router_info = value
        for decoder in self.decoders:
            decoder.return_router_info = value
            decoder.moe.return_router_info = value

    def forecast(self, X: torch.Tensor, context_window: int, prediction_length: int) -> torch.Tensor:
        X = X[:, -context_window:, :]
        return self.forward(X, prediction_length)

if __name__ == "__main__":
    X = torch.randn(1, 2, 100)
    decoder = TanaForecast(context_window=100, prediction_length=12)
    y = decoder(X)
    print(y)