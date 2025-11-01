import torch
import pandas as pd
from pathlib import Path
from src.model import TanaForecast
from src.trainer import TanaForecastTrainer, TimeSeriesDataset, Logger

class TrainingPipeline:
    """Training pipeline for the TanaForecast model."""
    def __init__(self, epochs: int, resume_training: bool = True) -> None:
        pass
        