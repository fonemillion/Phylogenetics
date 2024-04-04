import math

from torch_geometric.nn.dense.linear import Linear
import torch
from torch.nn import Sigmoid
from torch_geometric.nn import MessagePassing, TransformerConv, ResGatedGraphConv
from torch_geometric.nn import SAGEConv, GATv2Conv

class LinWave(MessagePassing):
    """
    MessagePassing Layer with no activation
    """
    def __init__(self, in_channels, out_channels, aggr = 'add'):
        super().__init__(aggr)
        self.out_channels = out_channels
        self.lin = torch.nn.Linear(out_channels + in_channels, out_channels)

    def forward(self, x, edge_index, nodeUp):
        maxUp = len(edge_index)
        x1 = torch.zeros((x.size()[0],self.out_channels))
        for i in range(maxUp):
            output = self.propagate(edge_index = edge_index[i], x = x1)
            x1[nodeUp[i], :] = self.lin(torch.hstack((output[nodeUp[i],:], x[nodeUp[i], :])))
        return x1

    def message(self, x_j):
        return x_j

class LayerWave(MessagePassing):
    """
    MessagePassing Layer with activation
    """
    def __init__(self, in_channels, out_channels, aggr = 'add'):
        super().__init__(aggr)
        self.out_channels = out_channels
        self.lin = torch.nn.Linear(out_channels + in_channels, out_channels)

    def forward(self, x, edge_index, nodeUp):
        maxUp = len(edge_index)
        x1 = torch.zeros((x.size()[0],self.out_channels))
        for i in range(maxUp):
            output = self.propagate(edge_index = edge_index[i], x = x1)
            x1[nodeUp[i], :] = torch.tanh(self.lin(torch.hstack((output[nodeUp[i],:], x[nodeUp[i], :]))))
        return x1

    def message(self, x_j):
        return x_j

class LinWaveRelu(MessagePassing):
    """
    MessagePassing Layer with activation
    """
    def __init__(self, in_channels, out_channels, aggr = 'add'):
        super().__init__(aggr)
        self.out_channels = out_channels
        self.lin = torch.nn.Linear(out_channels + in_channels, out_channels)

    def forward(self, x, edge_index, nodeUp):
        maxUp = len(edge_index)
        x1 = torch.zeros((x.size()[0],self.out_channels))
        for i in range(maxUp):
            output = self.propagate(edge_index = edge_index[i], x = x1)
            x1[nodeUp[i], :] = torch.relu(self.lin(torch.hstack((output[nodeUp[i],:], x[nodeUp[i], :]))))
        return x1

    def message(self, x_j):
        return x_j

class SageWave(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.mp = SAGEConv(in_channels + out_channels,out_channels)

    def forward(self, x, edge_index, nodeUp):
        maxUp = len(edge_index)
        x1 = torch.zeros((x.size()[0],self.out_channels))
        for i in range(maxUp):
            # output = self.propagate(edge_index = edge_index[i], x = x1)
            x1[nodeUp[i], :] = torch.relu(self.mp(torch.hstack((x, x1)), edge_index[i])[nodeUp[i], :])
        return x1


class GatWave(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr = 'add'):
        super().__init__(aggr)
        self.out_channels = out_channels
        self.mp = GATv2Conv(in_channels + out_channels,out_channels, aggr=aggr)

    def forward(self, x, edge_index, nodeUp):
        maxUp = len(edge_index)
        x1 = torch.zeros((x.size()[0],self.out_channels))
        for i in range(maxUp):
            # output = self.propagate(edge_index = edge_index[i], node_features = x1)
            # x1[nodeUp[i], :] = torch.tanh(self.mp(torch.hstack((x, x1)), edge_index[i])[nodeUp[i], :])
            x1[nodeUp[i], :] = torch.relu(self.mp(torch.hstack((x, x1)), edge_index[i])[nodeUp[i], :])
        return x1

class TransWave(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim = 0):
        super().__init__()
        self.out_channels = out_channels
        if edge_dim == 0:
            self.mp = TransformerConv(in_channels + out_channels,out_channels)
        else:
            self.mp = TransformerConv(in_channels + out_channels,out_channels, edge_dim=edge_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.mp.reset_parameters()

    def forward(self, x, edge_index, nodeUp, edge_attr = None):
        maxUp = len(edge_index)
        x1 = torch.zeros((x.size()[0],self.out_channels))

        if edge_attr == None:
            for i in range(maxUp):
                if int(x1[nodeUp[i], :].size()[0]) == 0:
                    continue
                x1[nodeUp[i], :] = torch.relu(self.mp(torch.hstack((x, x1)), edge_index[i])[nodeUp[i], :])
        else:
            for i in range(maxUp):
                if int(x1[nodeUp[i], :].size()[0]) == 0:
                    continue
                x1[nodeUp[i], :] = torch.relu(self.mp(torch.hstack((x, x1)), edge_index[i],edge_attr = edge_attr[i])[nodeUp[i], :])

        return x1

class cliqueLayer(MessagePassing):
    """
    MessagePassing Layer with activation
    """

    def __init__(self, in_channels, out_channels, edge_dim, aggr='add'):
        super().__init__(aggr)
        self.out_channels = out_channels
        self.mp = SAGEConv(in_channels + out_channels, out_channels, aggr='add')
        self.lin1 = torch.nn.Linear(in_channels + in_channels + edge_dim, out_channels)
        self.lin2 = torch.nn.Linear(in_channels + out_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        prop = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
        x1 = self.lin2(torch.hstack((x, prop)))

        return x1

    def message(self, x_j, x_i, edge_attr):
        x = torch.relu(self.lin1(torch.hstack((x_j, x_i, edge_attr))))
        return x


class cliqueLayerPlus(MessagePassing):
    """
    MessagePassing Layer with activation
    """

    def __init__(self, in_channels, out_channels, edge_dim, aggr='add'):
        super().__init__(aggr)
        self.out_channels = out_channels
        self.mp = SAGEConv(in_channels + out_channels, out_channels, aggr='add')
        self.lin1 = torch.nn.Linear(in_channels + in_channels + edge_dim, out_channels)
        self.lin2 = torch.nn.Linear(in_channels + out_channels, out_channels)
        self.lin3 = torch.nn.Linear(out_channels + out_channels + edge_dim, edge_dim)

    def forward(self, x, edge_index, edge_attr):
        prop = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
        x1 = self.lin2(torch.hstack((x, prop)))


        for i in range(edge_index.size()[1]):
            edge = edge_index[:,i]
            # print(x1[edge[0],:])

            edge_attr[i, :] = torch.relu(self.lin3(torch.hstack((x1[edge[0],:], x1[edge[1],:], edge_attr[i, :]))))
        # print(edge_index)

        return x1,edge_attr

    def message(self, x_j, x_i, edge_attr):
        x = torch.relu(self.lin1(torch.hstack((x_j, x_i, edge_attr))))
        return x


class ResGatedGraphConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim = None,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super(ResGatedGraphConv, self).__init__(**kwargs)

        # if edge_dim = None:


        self.lin = torch.nn.Linear(in_channels, out_channels)

        self.lin_key = Linear(edge_dim+in_channels, out_channels)
        self.lin_query = Linear(edge_dim+in_channels, out_channels)
        self.lin_value = Linear(edge_dim+in_channels, out_channels)
        self.act = torch.nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()


    def forward(self, x, edge_index, edge_attr):


        k, q, v = x, x, x

        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr,
                             size=None)

        out = out + self.lin(x)

        return out


    def message(self, k_i, q_j, v_j, edge_attr):

        k_i = self.lin_key(torch.cat([k_i, edge_attr], dim=-1))
        q_j = self.lin_query(torch.cat([q_j, edge_attr], dim=-1))
        v_j = self.lin_value(torch.cat([v_j, edge_attr], dim=-1))

        return self.act(k_i + q_j) * v_j