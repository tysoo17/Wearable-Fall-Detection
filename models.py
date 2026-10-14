"""3 loại mô hình phát hiện ngã: ngưỡng cố định, Random Forest, 1D-CNN."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

from common import features, magnitude


class ThresholdModel:
    """Baseline giống nhiều thiết bị đơn giản: gia tốc tổng vượt ngưỡng là ngã."""

    def fit(self, X, y, rate):
        peak = magnitude(X).max(axis=1)
        cands = np.linspace(1.5, 6, 91)
        # Chọn ngưỡng cân bằng độ nhạy và độ đặc hiệu (Youden) trên tập train.
        scores = [((peak[y == 1] > t).mean() + (peak[y == 0] <= t).mean()) for t in cands]
        self.t = float(cands[int(np.argmax(scores))])
        return self

    def predict_proba(self, X, rate):
        return (magnitude(X).max(axis=1) > self.t).astype(np.float32)


class RFModel:
    def fit(self, X, y, rate):
        self.clf = RandomForestClassifier(300, class_weight="balanced", n_jobs=-1, random_state=0)
        self.clf.fit(features(X, rate), y)
        return self

    def predict_proba(self, X, rate):
        return self.clf.predict_proba(features(X, rate))[:, 1]


class FallCNN(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        layers, c = [], in_ch
        for out in (16, 32, 64):
            layers += [nn.Conv1d(c, out, 5, padding=2), nn.BatchNorm1d(out), nn.ReLU(), nn.MaxPool1d(2)]
            c = out
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(c, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


class CNNModel:
    def __init__(self, input_mode, epochs=25):
        self.input_mode, self.epochs = input_mode, epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _prep(self, X):
        if self.input_mode == "mag":
            X = magnitude(X)[..., None]   # 1 kênh: không phụ thuộc hướng đeo thiết bị
        return torch.from_numpy(np.ascontiguousarray(X.transpose(0, 2, 1), dtype=np.float32))

    def fit(self, X, y, rate):
        torch.manual_seed(0)
        Xt, yt = self._prep(X), torch.from_numpy(y.astype(np.float32))
        self.model = FallCNN(Xt.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        pos_w = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1), device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        for _ in range(self.epochs):
            self.model.train()
            perm = torch.randperm(len(yt))
            for i in range(0, len(perm), 64):
                b = perm[i:i + 64]
                opt.zero_grad()
                loss = loss_fn(self.model(Xt[b].to(self.device)), yt[b].to(self.device))
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, X, rate):
        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model(self._prep(X).to(self.device))).cpu().numpy()
