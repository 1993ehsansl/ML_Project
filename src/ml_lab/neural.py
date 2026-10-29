"""PyTorch MLP and CNN architectures; training is performed in experiments.py."""
import torch
from torch import nn


class RegressionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 32),
                                 nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)


class DigitsCNN(nn.Module):
    """CNN for sklearn's 8x8 grayscale handwritten digits (not MNIST)."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                                      nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1),
                                      nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32*2*2, 64),
                                        nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


def train_torch(model, X, y, task='classification', epochs=15, lr=.001,
                batch_size=128, seed=42):
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch_x = torch.tensor(X, dtype=torch.float32)
    dtype = torch.long if task == 'classification' else torch.float32
    torch_y = torch.tensor(y, dtype=dtype)
    if task == 'regression':
        torch_y = torch_y.reshape(-1, 1)
    dataset = torch.utils.data.TensorDataset(torch_x, torch_y)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                         generator=generator)
    criterion = nn.CrossEntropyLoss() if task == 'classification' else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    model.train()
    for _ in range(epochs):
        total = 0.
        for inputs, targets in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total += loss.item()*len(inputs)
        history.append(total/len(dataset))
    model.eval()
    return model, history


def torch_predict(model, X, task='classification'):
    with torch.no_grad():
        outputs = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return np.argmax(outputs, axis=1) if task == 'classification' else outputs.ravel()


import numpy as np
