"""Experimental DBRL v4: endpoint-eligible dense context with local correction."""
import torch
from torch_geometric.data import Batch,Data
from src.models.hdeg.dbrl import DBRL

class ResidualContextDBRL(DBRL):
    def forward(self,x):
        self._validate_input(x)
        if x.ndim!=4 or x.shape[-1]!=6:
            raise ValueError('Residual-context DBRL requires six-feature inputs')
        h=self._encode_temporal_behavior(x)
        eligible=x[:,-1,:,1]==1
        graphs=[]
        for k in range(len(x)):
            edge=eligible[k,:,None]&eligible[k,None,:]
            edge=edge.clone();edge.fill_diagonal_(True)
            ix=edge.nonzero(as_tuple=False).t().contiguous()
            graphs.append(Data(x=h[k],edge_index=ix,num_nodes=h.shape[1]))
        graph=Batch.from_data_list(graphs)
        c=self.graph_reasoner(graph)
        local=self.graph_reasoner.gat.lin(graph.x).reshape_as(h)+self.graph_reasoner.gat.bias
        peer=eligible & (eligible.sum(dim=1,keepdim=True)>1)
        result=torch.where(peer.unsqueeze(-1),h+(c-local),h)
        if not torch.isfinite(result).all():raise ValueError('Nonfinite residual-context DBRL output')
        return result
