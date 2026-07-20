import torch
import torch.nn as nn
import numpy as np

class Graph:
    """
    Defines the NTU RGB+D 120 standard 25-joint skeletal graph topology.
    """
    def __init__(self):
        self.num_node = 25
        # 0-based index for Kinect v2 joints
        self.edges = [
            (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), 
            (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), 
            (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), 
            (22, 7), (23, 24), (24, 11)
        ]
        self.A = self.get_adjacency()
        
    def get_adjacency(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edges:
            A[i, j] = 1
            A[j, i] = 1
        A = A + np.eye(self.num_node) # Add self-loops
        
        # Symmetrical Normalization
        D = np.sum(A, axis=1)
        D_inv = np.diag(D**-0.5)
        A = D_inv @ A @ D_inv
        
        return torch.tensor(A, dtype=torch.float32).unsqueeze(0) # (1, V, V)


class SpatialGraphConv(nn.Module):
    """
    Spatial Graph Convolution layer.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x, A):
        # x: (N, C, T, V)
        x = self.conv(x)
        # A: (1, V, V) -> Matrix multiplication over vertices
        x = torch.einsum('nctv,kvw->nctw', (x, A))
        return x.contiguous()


class STGCNBlock(nn.Module):
    """
    Standard ST-GCN block: Spatial Graph Conv followed by Temporal Conv.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.sgc = SpatialGraphConv(in_channels, out_channels)
        # Temporal convolution over the window (kernel size 9)
        self.tgc = nn.Conv2d(out_channels, out_channels, (9, 1), (stride, 1), (4, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, A):
        x = self.relu(self.sgc(x, A))
        x = self.bn(self.tgc(x))
        x = self.relu(x)
        return x


class PGMT_STGCN(nn.Module):
    """
    Wraps the ST-GCN specifically for the PGMT Dual-Stream architecture.
    Strips the classification head to output the continuous latent vectors.
    """
    def __init__(self, in_channels=3, hidden_channels=256):
        super().__init__()
        self.graph = Graph()
        self.register_buffer('A', self.graph.A)
        
        self.l1 = STGCNBlock(in_channels, 64)
        self.l2 = STGCNBlock(64, 128)
        self.l3 = STGCNBlock(128, hidden_channels)

    def forward(self, x):
        # Input shape from DataLoader: (B, T, W, S, V, C)
        B, T, W, S, V, C = x.shape
        
        # Collapse B, T, and S into the batch dimension for standard ST-GCN processing
        # STGCN expects (N, C, Time, Vertices) -> (B*T*S, C, W, V)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous() 
        x = x.view(B * T * S, C, W, V)
        
        # Extract Spatial-Temporal Features
        x = self.l1(x, self.A)
        x = self.l2(x, self.A)
        x = self.l3(x, self.A) # -> (B*T*S, hidden_channels, W, V)
        
        # Global Average Pooling over the 25 physical nodes
        x = x.mean(dim=3) # -> (B*T*S, hidden_channels, W)
        
        # Reshape and Permute to match TemporalPooler expectations: (B, T, S, W, D)
        x = x.view(B, T, S, -1, W) 
        x = x.permute(0, 1, 2, 4, 3).contiguous() 
        
        return x