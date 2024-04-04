

import torch
from torch.nn import Linear, ModuleList, ReLU
from torch_geometric.nn import TransformerConv, GraphNorm, GATv2Conv, InstanceNorm
from waveLayers import TransWave, ResGatedGraphConv, GatWave, SageWave


class Clique(torch.nn.Module):
    def __init__(self, xf=6, ef=12, out=2, hidden_channels=20, num_layers=3, l_type=None, l_act='relu', changing_edge_feature = True):
        super(Clique, self).__init__()
        self.num_layers = num_layers
        self.cef = changing_edge_feature
        if l_type == None:
            raise Exception("no type")

        self.n = GraphNorm(xf)
        self.n2 = GraphNorm(ef)

        self.convs = ModuleList()
        for _ in range(num_layers):
            if l_type == "TransCq":
                self.convs.append(TransformerConv(hidden_channels, hidden_channels, edge_dim=hidden_channels))
            if l_type == "GATCq":
                self.convs.append(GATv2Conv(hidden_channels, hidden_channels, edge_dim=hidden_channels))
            if l_type == "ResGatedCq":
                self.convs.append(ResGatedGraphConv(hidden_channels, hidden_channels, edge_dim=hidden_channels))
        self.act = None
        if l_act == 'relu':
            self.act = ReLU()

        self.lin_in = Linear(xf, hidden_channels)
        self.lin_edge = Linear(ef, hidden_channels)
        self.lin_Out = Linear(hidden_channels, out)

        if self.cef:
            self.convs_edge = ModuleList()
            for _ in range(num_layers):
                self.convs_edge.append(Linear(3*hidden_channels, hidden_channels))

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()
        if self.cef:
            for conv in self.convs_edge:
                conv.reset_parameters()
        if hasattr(self.n, 'reset_parameters'):
            self.n.reset_parameters()
        if hasattr(self.n2, 'reset_parameters'):
            self.n2.reset_parameters()
        self.lin_in.reset_parameters()
        self.lin_edge.reset_parameters()
        self.lin_Out.reset_parameters()

    def forward(self, data):
        x = data.x
        z = data.edge_attr

        row, col = data.edge_index
        edge_batch = data.batch[row]

        x = self.n(x,batch = data.batch)
        z = self.n2(z,batch = edge_batch)

        x = torch.relu(self.lin_in(x))
        z = torch.relu(self.lin_edge(z))

        row, col = data.edge_index

        for l in range(self.num_layers):
            x = self.convs[l](x, data.edge_index, z)
            if self.act != None:
                x = self.act(x)
            if self.cef:
                if l != self.num_layers-1:
                    z = self.act(self.convs_edge[l](torch.cat([x[row], x[col], z], dim=-1)))
        x = x[data.pickAble]
        x = self.lin_Out(x)

        return x


class DAG(torch.nn.Module):
    def __init__(self, num_features=10, num_classes=2, hidden_channels=20, num_layers=3, l_type='relu'):
        super(DAG, self).__init__()
        self.num_layers = num_layers

        if l_type == None:
            raise Exception("no type")

        self.n = InstanceNorm(num_features)

        self.convs = ModuleList()
        for _ in range(self.num_layers):
            if l_type == "TransLay":
                self.convs.append(TransWave(hidden_channels, hidden_channels))
                self.convs.append(TransWave(hidden_channels, hidden_channels))
            if l_type == "GatDAG":
                self.convs.append(GatWave(hidden_channels, hidden_channels))
                self.convs.append(GatWave(hidden_channels, hidden_channels))
            if l_type == "SageDAG":
                self.convs.append(SageWave(hidden_channels, hidden_channels))
                self.convs.append(SageWave(hidden_channels, hidden_channels))

        self.lin_in = Linear(num_features, hidden_channels)
        self.lin_Out = Linear(hidden_channels, num_classes)

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()

        if hasattr(self.n, 'reset_parameters'):
            self.n.reset_parameters()
        self.lin_in.reset_parameters()
        self.lin_Out.reset_parameters()

    def forward(self, data):
        x = data.x

        x = self.n(x,batch = data.batch)

        x = torch.relu(self.lin_in(x))

        for l in range(self.num_layers*2):
            if l % 2 == 0:
                x = self.convs[l](x, data.edge_index_flowEdgeUp, data.nodeMaskUp)
            else:
                x = self.convs[l](x, data.edge_index_flowEdgeDown, data.nodeMaskDown)
        x = x[data.pickAble]
        x = self.lin_Out(x)

        return x

