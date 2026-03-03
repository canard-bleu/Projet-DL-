"""NICE model utilities adapted from the TFD notebook.

This module exposes training/evaluation helpers used by the Streamlit app.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor


def get_device():
    """Détermine meilleur device disponible """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_mnist_loaders(batch_size=64, valid_ratio=0.1, seed=42):
    """Récupérer dataset et loader pour l'entraînement, la validation et le test sur MNIST"""
    transform = ToTensor()

    train_and_valid = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    n_total = len(train_and_valid)
    n_valid = int(n_total * valid_ratio)
    n_train = n_total - n_valid
    train_subset, valid_subset = random_split(
        train_and_valid, [n_train, n_valid], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader


def log_logistic_distrib_soft(x):
    """Log-distribution logistique, version stable avec softplus du package nn.functional """
    return -F.softplus(x) - F.softplus(-x)


class ModeleM(nn.Module):
    """MLP des couches additives"""

    def __init__(self, in_dim, hid_dim, out_dim, num_hid_lay):
        super().__init__()
        self.num_hid_lay = num_hid_lay
        self.lay_in = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.ReLU())
        self.lay_out = nn.Linear(hid_dim, out_dim)
        self.lay_hid = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU())

    def forward(self, x):
        out = self.lay_in(x)
        for _ in range(self.num_hid_lay - 1):
            out = self.lay_hid(out)
        return self.lay_out(out)


class NICE(nn.Module):
    """Modèle NICE complet, avec couches additives et scaling exponentiel """

    def __init__(self, in_dim=392, hid_dim=1000, out_dim=392, num_hid_lay=5, nb_add=4):
        super().__init__()
        self.nb_add = nb_add
        self.modeles = nn.ModuleList(
            [ModeleM(in_dim, hid_dim, out_dim, num_hid_lay) for _ in range(self.nb_add)]
        )
        self.s = nn.Parameter(torch.zeros(int(2 * in_dim)))

    def _scale(self):
        """Borne s pour empêcher explosion pendant l'entraînement"""
        return 5.0 * torch.tanh(self.s)

    def forward(self, x):
        for k in range(self.nb_add):
            x1, x2 = torch.chunk(x, 2, dim=1)
            x2 = x2 + self.modeles[k](x1)
            x = torch.cat([x2, x1], dim=1)

        s = self._scale()
        z = torch.exp(s) * x

        log_lik = torch.zeros(1)
        log_lik = torch.sum(log_logistic_distrib_soft(z), dim=1).mean() + torch.sum(s)
        loss = -log_lik
        return z, loss

    def inverse(self, h):
        s = self._scale()
        h = torch.exp(-s) * h

        for k in range(self.nb_add):
            h1, h2 = torch.chunk(h, 2, dim=1)
            h1 = h1 - self.modeles[self.nb_add - 1 - k](h2)
            h = torch.cat([h2, h1], dim=1)

        return h


def checkpoint_paths(hid_dim, num_hid_lay, nb_add, lr):
    """Return checkpoint and history paths for a NICE configuration."""
    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")
    stem = (
        "saved_models/nice_mnist_hid="
        f"{hid_dim}_layers={num_hid_lay}_add={nb_add}_lr={lr:.0e}"
    )
    return stem + ".pth", stem + "_history.pt"


def train_loop(dataloader, model, optimizer, epochs, device, progress_callback=None):
    """Boucle d'entraînement du modèle"""
    flatten = nn.Flatten()
    history = []
    model.train()
    for epoch_idx in range(epochs):
        epoch_loss = 0.0
        size = len(dataloader)
        for X, _ in dataloader:
            X = flatten(X.to(device))
            noise = torch.rand_like(X)
            X = (255 * X + noise) / 256

            optimizer.zero_grad()
            _, loss = model.forward(X)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
        mean_epoch_loss = epoch_loss / size
        history.append(mean_epoch_loss)
        if progress_callback is not None:
            progress_callback(epoch_idx + 1, epochs, mean_epoch_loss)
    return history


def evaluate_log_likelihood(model, dataloader, device):
    """Boucle de test : calcule la loglikelihood sur le dataset fourni"""
    was_training = model.training
    model.eval()
    flatten = nn.Flatten()

    total_log_lik = 0.0
    size = len(dataloader)

    with torch.no_grad():
        for X, _ in dataloader:
            X = flatten(X.to(device))

            noise = torch.rand_like(X)
            X = (255 * X + noise) / 256
            _, loss = model.forward(X)
            total_log_lik -= loss.item()


    if was_training:
        model.train()

    avg_log_lik = total_log_lik / size
    return avg_log_lik


def sample_logistic(shape, device):
    """Génère des éléments issus d'une distribution logistique """
    u = torch.rand(shape, device=device)
    eps = 1e-7 #sécurité contre log(0)
    u = torch.clamp(u, eps, 1 - eps)
    return torch.log(u) - torch.log(1 - u)


def generate_samples(model, device, n_samples=10):
    """Génère des images 'de type MNIST' à l'aide de la fonction inverse du modèle NICE"""
    model.eval()
    with torch.no_grad():
        z = sample_logistic((n_samples, 784), device)
        x = model.inverse(z).view(-1, 28, 28)
    return torch.clamp(x, 0.0, 1.0).detach().cpu()
