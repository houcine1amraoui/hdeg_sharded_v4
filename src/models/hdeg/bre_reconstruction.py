"""BRE reconstruction model. No HBF parameters or future-window branch."""
import torch
from torch import nn
from src.models.hdeg.residual_context_dbrl import ResidualContextDBRL
from src.models.hdeg.bse import BehavioralStateEstimator
from src.models.hdeg.ebrl import EcosystemBehavioralRepresentationLearner
from src.models.hdeg.bre_hierarchy import BREInteractionLearner
from src.models.hdeg.reconstruction import HierarchicalReconstructor


class BREReconstructionModel(nn.Module):
    def __init__(self, config, semantics):
        super().__init__()
        h = config['hdeg']; cfg = h['dbrl']; d = cfg.get('embedding_dim', 64)
        n = len(semantics['device_ids']); k = len(semantics['state_ids'])
        if cfg.get('graph_heads', 1) != 1:
            raise ValueError('Residual-context DBRL requires one graph head')
        self.state_ids = tuple(semantics['state_ids'])
        self.num_devices = n; self.num_states = k; self.embedding_dim = d
        self.window_size = config['preprocessing']['window_size']
        self.dbrl = ResidualContextDBRL(hidden_dim=cfg.get('hidden_dim', 64), embedding_dim=d,
            gru_layers=cfg.get('gru_layers', 1), dropout=0., graph_top_k=cfg.get('graph_top_k', 15),
            graph_self_loops=True, graph_symmetric=True, graph_heads=1, graph_dropout=0., input_dim=6)
        self.bse = BehavioralStateEstimator(k, d, num_heads=h['bse'].get('num_heads', 1))
        self.bil = BREInteractionLearner(d, k, semantics['edge_index'])
        self.ebrl = EcosystemBehavioralRepresentationLearner(d, num_states=k)
        self.register_buffer('compatibility_mask', torch.tensor(semantics['compatibility_mask'], dtype=torch.bool))
        self.reconstructor = HierarchicalReconstructor(n, k, d, self.window_size,
            h.get('reconstruction', {}).get('hidden_dim', 64))

    def encode(self, x):
        if x.ndim != 4 or x.shape[1:] != (self.window_size, self.num_devices, 6):
            raise ValueError('Expected B,W,N,6 inputs')
        z = self.dbrl(x); s = self.bse(z, self.compatibility_mask); st = self.bil(s)
        return dict(Z=z, S=s, S_tilde=st, g=self.ebrl(st))

    def forward(self, x):
        representations = self.encode(x)
        return dict(representations=representations, reconstructions=self.reconstructor(representations))

    def import_encoder_stage(self, checkpoint):
        if checkpoint.get('protocol') != 'bre_reconstruction_stage_v1' or checkpoint.get('hbf_steps') != 0:
            raise ValueError('Expected encoder-stage checkpoint; forecasting checkpoint is not compatible')
        state = {k: v for k, v in checkpoint['model'].items() if not k.startswith('hbf.')}
        state.update({'reconstructor.' + k: v for k, v in checkpoint['decoders'].items()})
        self.load_state_dict(state, strict=True)
