"""BRE adapter; original CU modules and topology remain untouched."""
import torch
from torch import nn
from src.models.hdeg.residual_context_dbrl import ResidualContextDBRL
from src.models.hdeg.bse import BehavioralStateEstimator
from src.models.hdeg.ebrl import EcosystemBehavioralRepresentationLearner
from src.models.hdeg.hbf import HierarchicalBehavioralForecaster

LEVELS=('Z','S','S_tilde','g')


class BREInteractionLearner(nn.Module):
    def __init__(self, embedding_dim, num_states, edge_index):
        super().__init__();self.num_states=num_states;self.embedding_dim=embedding_dim
        edges=torch.as_tensor(edge_index,dtype=torch.long)
        if edges.ndim!=2 or edges.shape[0]!=2 or (edges<0).any() or (edges>=num_states).any():raise ValueError('Invalid BRE interaction graph')
        if (edges[0]==edges[1]).any():raise ValueError('No semantic self-loops')
        if len(set(map(tuple,edges.t().tolist())))!=edges.shape[1]:raise ValueError('Duplicate edges')
        self.register_buffer('edge_index',edges)
        mask=torch.zeros(num_states,num_states,dtype=torch.bool);mask[edges[1],edges[0]]=True
        self.register_buffer('incoming_mask',mask)
        self.state_projection=nn.Linear(embedding_dim,embedding_dim,bias=False)
        self.attention_vector=nn.Parameter(torch.empty(2*embedding_dim))
        self.context_refinement=nn.Linear(embedding_dim,embedding_dim)
        nn.init.xavier_uniform_(self.state_projection.weight);nn.init.xavier_uniform_(self.attention_vector[None])
        nn.init.xavier_uniform_(self.context_refinement.weight);nn.init.zeros_(self.context_refinement.bias)

    def forward(self,s):
        if s.ndim!=3 or s.shape[1:]!=(self.num_states,self.embedding_dim) or not torch.isfinite(s).all():raise ValueError('Invalid BIL input')
        h=self.state_projection(s);source,target=self.edge_index
        scores=torch.nn.functional.leaky_relu((torch.cat([h[:,target],h[:,source]],-1)*self.attention_vector).sum(-1),.2)
        matrix=s.new_full((len(s),self.num_states,self.num_states),float('-inf'));matrix[:,target,source]=scores
        incoming=self.incoming_mask.any(-1)
        # Safe softmax for isolated states avoids NaN in forward AND backward.
        matrix=torch.where(incoming[None,:,None],matrix,torch.zeros_like(matrix))
        attention=torch.softmax(matrix,-1)*self.incoming_mask[None]
        refined=self.context_refinement(attention@h)
        return s+torch.where(incoming[None,:,None],refined,torch.zeros_like(refined))


class BREHierarchy(nn.Module):
    def __init__(self, config, semantics):
        super().__init__();h=config['hdeg'];cfg=h['dbrl'];d=cfg.get('embedding_dim',64)
        n=len(semantics['device_ids']);k=len(semantics['state_ids'])
        if cfg.get('graph_heads',1)!=1:raise ValueError('Residual-context DBRL contract requires one graph head')
        self.num_devices=n;self.num_states=k;self.embedding_dim=d
        self.dbrl=ResidualContextDBRL(hidden_dim=cfg.get('hidden_dim',64),embedding_dim=d,gru_layers=cfg.get('gru_layers',1),
            dropout=0.,graph_top_k=cfg.get('graph_top_k',15),graph_self_loops=True,graph_symmetric=True,
            graph_heads=1,graph_dropout=0.,input_dim=6)
        self.bse=BehavioralStateEstimator(k,d,num_heads=h['bse'].get('num_heads',1))
        self.bil=BREInteractionLearner(d,k,semantics['edge_index'])
        self.ebrl=EcosystemBehavioralRepresentationLearner(d,num_states=k)
        self.hbf=HierarchicalBehavioralForecaster(num_devices=n,num_states=k,embedding_dim=d,dynamics_hidden_dim=h['hbf'].get('dynamics_hidden_dim',128))
        self.register_buffer('compatibility_mask',torch.tensor(semantics['compatibility_mask'],dtype=torch.bool))
        # Explicit v4 residual HBF convention, not the original absolute-output convention.
        for name in ('device_prediction_head','state_prediction_head','context_prediction_head','global_prediction_head'):
            head=getattr(self.hbf,name);nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def encode(self,x):
        if x.ndim!=4 or x.shape[2:]!=(self.num_devices,6):raise ValueError('Expected B,W,N,6 input')
        z=self.dbrl(x);s=self.bse(z,self.compatibility_mask);st=self.bil(s)
        return dict(Z=z,S=s,S_tilde=st,g=self.ebrl(st))

    def predict(self,current):
        delta=self.hbf(*(current[k] for k in LEVELS))
        return {k:current[k]+delta[k] for k in LEVELS}

    def forward(self,x):
        current=self.encode(x)
        return dict(representations=current,predictions=self.predict(current))
