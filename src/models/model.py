import torch
import torch.nn as nn
import torch.nn.functional as F
import math

PATCH_SIZE = 24
STRIDE = 12
D_OUT = 64

class Patch(nn.Module):
    """
    Cut the dataset into N patches.
    """
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

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: torch.Tensor, the input tensor of shape (B, N, D)
        """
        B, N, D = X.shape

        q = self.q(X)
        k = self.k(X)
        v = self.v(X)

        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_model)

        mask = torch.tril(torch.ones(N, N, device=X.device)) # this function stores the mask on CPU by default.
        mask = mask.masked_fill(mask == 0, float('-inf'))

        attention_scores = attention_scores + mask
        attention_weights = F.softmax(attention_scores, dim=-1)

        return self.out(attention_weights @ v)

class Encoder(nn.Module):
    def __init__(self, X: torch.Tensor, context_window: int) -> None:
        super().__init__()
        self.B, self.C, self.L = X.shape
        self.patch = Patch(
            X=X,
            context_window=context_window,
            patch_size=PATCH_SIZE,
            stride=STRIDE,
            d_out=D_OUT
        )
        self.layer_norm_1 = nn.LayerNorm(D_OUT)
        self.attention = CausalSelfAttention(d_model=D_OUT)
        self.layer_norm_2 = nn.LayerNorm(D_OUT)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: torch.Tensor, the input tensor of shape (B, C, L)
        """
        X = self.patch(X)
        X = self.layer_norm_1(X)
        X = self.attention(X)
        return self.layer_norm_2(X)
if __name__ == "__main__":
    X = torch.randn(1, 2, 100)
    print(X)
    encoder = Encoder(X, context_window=100)
    print(encoder(X))