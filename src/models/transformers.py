from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class SimpleTransformerRegressor(nn.Module):
    # Simple transformer architecture for regression tasks
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()  # Initialize parent class
        self.input_proj = nn.Linear(input_dim, d_model)  # Project input to model dimension
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=dropout, batch_first=True)  # Create encoder layer
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # Stack encoder layers
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))  # Output head for regression

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Forward pass through the transformer architecture
        # Reshape input as single-token sequence for transformer
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        z = self.encoder(x)
        y = self.head(z.squeeze(1))
        return y.squeeze(-1)


def _torch_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # Calculate metrics for PyTorch model evaluation
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    err = np.abs(y_true - y_pred)
    acc20 = float((err <= 20.0).mean())
    acc10 = float((err <= 10.0).mean())
    acc5 = float((err <= 5.0).mean())
    acc = acc10
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "accuracy": acc, "accuracy5": acc5, "accuracy10": acc10, "accuracy20": acc20}


def train_eval_simple_transformer(X, y, epochs: int = 10, batch_size: int = 1024, lr: float = 1e-3, random_state: int = 42):
    # Train and evaluate the transformer regression model
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    X_train, X_test, y_train, y_test = train_test_split(X.values.astype(np.float32), y.values.astype(np.float32), test_size=0.2, random_state=random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    model = SimpleTransformerRegressor(input_dim=X_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    def to_batches(A, b, bs):
        # Create batches for training
        n = len(A)
        for i in range(0, n, bs):
            yield A[i:i+bs], b[i:i+bs]

    # Training loop
        model.train()
        for xb, yb in to_batches(X_train, y_train, batch_size):
            xb_t = torch.from_numpy(xb).to(device)
            yb_t = torch.from_numpy(yb).to(device)
            opt.zero_grad()
            pred = model(xb_t)
            loss = loss_fn(pred, yb_t)
            loss.backward()
            opt.step()

    # Evaluation phase
    # Disable gradient computation for inference
        y_pred = []
        for xb, _ in to_batches(X_test, y_test, batch_size):
            xb_t = torch.from_numpy(xb).to(device)
            pred = model(xb_t).detach().cpu().numpy()
            y_pred.append(pred)
        y_pred = np.concatenate(y_pred)

    return model, _torch_metrics(y_test, y_pred)



