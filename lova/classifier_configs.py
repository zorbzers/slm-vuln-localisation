# File:         lova/classifier_configs.py
# Author:       Lea Button
# Date:         25-09-2025
# Description:  Configuration for various classifiers used in LoVA.

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

class LSTMClassifier(nn.Module):
    """ LSTM-based classifier for sequence data. """
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.0, bidirectional=False):
        """
        Initialize the LSTM classifier.
        
        Parameters
        ----------
        input_dim : int
            Dimension of input features.
        hidden_dim : int, optional
            Dimension of LSTM hidden states. Default is 128.
        num_layers : int, optional
            Number of LSTM layers. Default is 1.
        dropout : float, optional
            Dropout rate between LSTM layers. Default is 0.0.
        bidirectional : bool, optional
            Whether to use a bidirectional LSTM. Default is False.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.fc(out).squeeze(-1)
        return logits
    

class MLPClassifier(nn.Module):
    """Simple feedforward neural network classifier."""
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        """
        Initialize the MLP classifier.
        
        Parameters
        ----------
        input_dim : int
            Dimension of input features.
        hidden_dim : int, optional
            Dimension of hidden layer. Default is 256.
        dropout : float, optional
            Dropout rate. Default is 0.3.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        logits = self.net(x)
        return logits.squeeze(-1)  


CLASSIFIER_CONFIGS = {
    "LSTM_small": lambda input_dim: (
        LSTMClassifier(input_dim, hidden_dim=64, num_layers=1, dropout=0.1, bidirectional=False),
        torch.optim.Adam,
        {"lr": 1e-3},
    ),
    "LSTM_moderate": lambda input_dim: (
        LSTMClassifier(input_dim, hidden_dim=128, num_layers=1, dropout=0.2, bidirectional=False),
        torch.optim.Adam,
        {"lr": 1e-3},
    ),
    "LSTM_large": lambda input_dim: (
        LSTMClassifier(input_dim, hidden_dim=256, num_layers=2, dropout=0.3, bidirectional=False),
        torch.optim.Adam,
        {"lr": 5e-4},
    ),
    "BiLSTM_small": lambda input_dim: (
        LSTMClassifier(input_dim, hidden_dim=64, num_layers=1, dropout=0.1, bidirectional=True),
        torch.optim.Adam,
        {"lr": 1e-3},
    ),
    "BiLSTM_moderate": lambda input_dim: (
        LSTMClassifier(input_dim, hidden_dim=128, num_layers=1, dropout=0.2, bidirectional=True),
        torch.optim.Adam,
        {"lr": 1e-3},
    ),
    "BiLSTM_large": lambda input_dim: (
        LSTMClassifier(input_dim, hidden_dim=256, num_layers=2, dropout=0.3, bidirectional=True),
        torch.optim.Adam,
        {"lr": 5e-4},
    ),
    "LogReg_default": lambda input_dim: (
        LogisticRegression(C=1.0, penalty="l2", max_iter=1000),
        None,
        None,
    ),
    "LogReg_lowC": lambda input_dim: (
        LogisticRegression(C=0.1, penalty="l2", max_iter=1000),
        None,
        None,
    ),
    "LogReg_l1": lambda input_dim: (
        LogisticRegression(C=1.0, penalty="l1", solver="liblinear", max_iter=1000),
        None,
        None,
    ),
    "XGB_small": lambda input_dim: (
        XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, n_jobs=-1, use_label_encoder=False, eval_metric="logloss"),
        None,
        None,
    ),
    "XGB_moderate": lambda input_dim: (
        XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, n_jobs=-1, use_label_encoder=False, eval_metric="logloss"),
        None,
        None,
    ),
    "XGB_large": lambda input_dim: (
        XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, n_jobs=-1, use_label_encoder=False, eval_metric="logloss"),
        None,
        None,
    ),
    # "MLP_small": lambda input_dim: (
    #     MLPClassifier(input_dim, hidden_dim=64, dropout=0.1),
    #     torch.optim.Adam,
    #     {"lr": 1e-3},
    # ),
    # "MLP_moderate": lambda input_dim: (
    #     MLPClassifier(input_dim, hidden_dim=128, dropout=0.2),
    #     torch.optim.Adam,
    #     {"lr": 1e-3},
    # ),
    # "MLP_large": lambda input_dim: (
    #     MLPClassifier(input_dim, hidden_dim=256, dropout=0.3),
    #     torch.optim.Adam,
    #     {"lr": 5e-4},
    # )
}
