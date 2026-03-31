import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class FlowConv(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        kernel_size: int = 3,
    ):
        Conv = getattr(nn, "Conv%dd" % spatial_dims)
        conv = Conv(
            in_channels,
            spatial_dims,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        conv.weight = nn.Parameter(Normal(0, 1e-5).sample(conv.weight.shape))
        super().__init__(conv)
        
class LearnablePositionEmbedding(nn.Module):
    def __init__(self,
                 embed_dim: int):
        super().__init__()
        self.mlps = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
             x: (B,)
        Returns:
            (B, embed_dim)
        '''
        return self.mlps(x[:, None])


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal Position Embeddings for Time.
    """

    def __init__(self,
                 embed_dim: int,
                 max_periods: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_periods = max_periods

    def forward(self, x: torch.Tensor):
        '''
        Args:
            x (torch.Tensor):  time indices, float, shape (B,)

        Returns:
            embeddings (torch.Tensor): (B, embed_dim)
        '''
        indices = torch.arange(0, self.embed_dim // 2).float().to(x.device)  # (embed_dim//2,)
        indices = torch.pow(self.max_periods, -2 * indices / self.embed_dim)
        embeddings = torch.einsum('b,d->bd', x, indices)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)  # (B, embed_dim)
        return embeddings