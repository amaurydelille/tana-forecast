import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from typing import Dict, List, Optional
from src.trainer import TimeSeriesDataset
import numpy as np

def plot_training_history(history: Dict[str, List[float]], height: int = 500, width: int = 1200) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Training History', 'Learning Rate Schedule')
    )
    
    fig.add_trace(
        go.Scatter(y=history['train_loss'], name='Train Loss', mode='lines'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=history['val_loss'], name='Val Loss', mode='lines'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(y=history['learning_rates'], name='Learning Rate', mode='lines', showlegend=False),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Loss (MSE)", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Learning Rate", row=1, col=2)
    
    fig.update_layout(height=height, width=width, showlegend=True)
    
    return fig

def plot_forecast(
    context: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    dataset: TimeSeriesDataset,
    title: str = "Time Series Forecast",
    height: int = 500,
    width: int = 1000,
    feature_idx: int = 0,
    feature_name: str = "Value"
) -> go.Figure:
    context_denorm = dataset.denormalize(context, is_target=False)
    target_denorm = dataset.denormalize(target, is_target=True)
    prediction_denorm = dataset.denormalize(prediction, is_target=True)
    
    context_len = context.shape[1]
    
    if target.dim() == 2:
        prediction_len = target.shape[1]
        target_values = target_denorm[feature_idx].cpu().numpy()
    else:
        prediction_len = target.shape[0]
        target_values = target_denorm.cpu().numpy()
    
    if prediction.dim() == 2:
        prediction_values = prediction_denorm[feature_idx].cpu().numpy()
    else:
        prediction_values = prediction_denorm.cpu().numpy()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(context_len)),
        y=context_denorm[feature_idx].cpu().numpy(),
        mode='lines',
        name='Historical',
        line=dict(width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(context_len, context_len + prediction_len)),
        y=target_values,
        mode='lines+markers',
        name='Actual Future',
        line=dict(width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(context_len, context_len + prediction_len)),
        y=prediction_values,
        mode='lines+markers',
        name='Predicted Future',
        line=dict(width=2, dash='dash'),
        marker=dict(size=8, symbol='square')
    ))
    
    fig.add_vline(
        x=context_len, 
        line_dash="dot", 
        line_color="red", 
        opacity=0.5, 
        annotation_text="Prediction Start"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='Time Steps',
        yaxis_title=feature_name,
        height=height,
        width=width,
        showlegend=True
    )
    
    return fig

def compute_metrics(target: torch.Tensor, prediction: torch.Tensor) -> Dict[str, float]:
    mse = ((target - prediction) ** 2).mean().item()
    mae = (target - prediction).abs().mean().item()
    rmse = mse ** 0.5
    mape = ((target - prediction).abs() / target.abs()).mean().item()
    smape = ((target - prediction).abs() / ((target.abs() + prediction.abs()) / 2)).mean().item()
    r2 = (1 - ((target - prediction) ** 2).sum() / ((target - target.mean()) ** 2).sum()).item()
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'smape': smape
    }

def print_metrics(metrics: Dict[str, float]) -> None:
    print(f"MSE:  {metrics['mse']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAPE: {metrics['mape']:.4f}")
    print(f"SMAPE: {metrics['smape']:.4f}")

