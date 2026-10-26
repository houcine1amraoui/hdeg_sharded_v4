"""Input-space hierarchical reconstruction, objective and deterministic evidence."""
import torch
from torch import nn

LEVELS = ('Z', 'S', 'S_tilde', 'g')


def weights_tensor(weights=None):
    weights = dict.fromkeys(LEVELS, 1.) if weights is None else weights
    if set(weights) != set(LEVELS):
        raise ValueError('Weights must specify Z, S, S_tilde, g exactly')
    if any(isinstance(weights[k], bool) or not isinstance(weights[k], (float, int)) for k in LEVELS):
        raise ValueError('Weights must be numeric')
    w = torch.tensor([weights[k] for k in LEVELS], dtype=torch.float32)
    if not torch.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('Weights must be finite, nonnegative, with positive total')
    return w / w.sum()


class HierarchicalReconstructor(nn.Module):
    """One decoder per level. All four reconstruct the SAME normalized X window."""
    def __init__(self, num_devices, num_states, embedding_dim, window_size, hidden_dim=64):
        super().__init__()
        self.window_size = window_size
        self.num_devices = num_devices
        sizes = (num_devices * embedding_dim, num_states * embedding_dim,
                 num_states * embedding_dim, embedding_dim)
        self.heads = nn.ModuleDict({k: nn.Sequential(nn.Flatten(1), nn.Linear(size, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, window_size * num_devices)) for k, size in zip(LEVELS, sizes)})

    def forward(self, representations):
        return {k: self.heads[k](representations[k]).reshape(-1, self.window_size, self.num_devices) for k in LEVELS}


def reconstruction_errors(observed, status, reconstructed):
    """Mask nonnumeric targets. Reject unscorable windows instead of zero anomalies."""
    if observed.ndim != 3 or status.shape != observed.shape:
        raise ValueError('Expected aligned B,W,N values and status')
    if not torch.isfinite(observed).all() or ((status < 0) | (status > 4)).any():
        raise ValueError('Invalid values/status')
    valid = status == 0
    counts = valid.sum((1, 2))
    if (counts == 0).any():
        raise ValueError('A window has no valid reconstruction targets')
    if set(reconstructed) != set(LEVELS):
        raise ValueError('Reconstruction levels differ')
    per_level, per_channel = {}, {}
    channel_counts = valid.sum(1)
    for k in LEVELS:
        pred = reconstructed[k]
        if pred.shape != observed.shape or not torch.isfinite(pred).all():
            raise ValueError('Invalid reconstruction: ' + k)
        squared = torch.where(valid, (pred - observed).square(), torch.zeros_like(pred))
        per_level[k] = squared.sum((1, 2)) / counts
        # Zero placeholders are accompanied by counts; never interpret absent channels as normal.
        per_channel[k] = squared.sum(1) / channel_counts.clamp_min(1)
    return dict(levels=per_level, channels=per_channel, valid_counts=counts, channel_counts=channel_counts)


class ReconstructionObjective(nn.Module):
    def __init__(self, weights=None):
        super().__init__(); self.register_buffer('weights', weights_tensor(weights))

    def forward(self, observed, status, reconstructed):
        errors = reconstruction_errors(observed, status, reconstructed)
        levels = {k: errors['levels'][k].mean() for k in LEVELS}
        loss = sum(self.weights[i] * levels[k] for i, k in enumerate(LEVELS))
        return loss, levels


class ReconstructionAnomalyInference(nn.Module):
    """MBAI reconstruction counterpart: four input-space errors, fixed fusion."""
    def __init__(self, weights=None):
        super().__init__(); self.register_buffer('weights', weights_tensor(weights))

    def forward(self, observed, status, reconstructed):
        errors = reconstruction_errors(observed, status, reconstructed)
        errors['score'] = sum(self.weights[i] * errors['levels'][k] for i, k in enumerate(LEVELS))
        return errors
