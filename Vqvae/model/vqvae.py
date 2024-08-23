import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VectorQuantizer(nn.Module):
    def __init__(self, num_embed, embed_dim, beta):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embed_dim
        self._num_embeddings = num_embed

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = beta

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        # inputs = inputs.contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embed, embed_dim, beta, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embed_dim
        self._num_embeddings = num_embed

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = beta

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embed))
        self._ema_w = nn.Parameter(torch.Tensor(num_embed, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        # inputs = inputs.contiguous()
        input_shape = inputs.shape
        ##print(f"input_shape: {input_shape}")

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        ###print(f"encoding_indices: {encoding_indices.shape}")
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encoding_indices.view(-1, H, W)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualStack(nn.Module):
    def __init__(self, dim, num_residual=2):
        super(ResidualStack, self).__init__()
        self.num_residual = num_residual
        self.layers = nn.ModuleList([ResBlock(dim) for _ in range(num_residual)])

    def forward(self, x):
        for i in range(self.num_residual):
            x = self.layers[i](x)
        return x


class VectorQuantizedVAE(nn.Module):
    def __init__(self, inc, num_embed=256, embed_dim=128, num_residual=2, beta=0.25, decay=0.99):
        super(VectorQuantizedVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(inc, embed_dim, 4, 2, 1, bias=False),

            nn.BatchNorm2d(embed_dim),
            nn.ReLU(True),
            nn.Conv2d(embed_dim, embed_dim, 4, 2, 1, bias=False),

            nn.BatchNorm2d(embed_dim),
            nn.ReLU(True),
            nn.Conv2d(embed_dim, embed_dim, 4, 2, 1, bias=False),

            # nn.BatchNorm2d(embed_dim),
            # nn.ReLU(True),
            # nn.Conv2d(embed_dim, embed_dim, (4, 4), (1, 2), 1, bias=False),

            ResidualStack(embed_dim, num_residual)
        )

        self.codebook = VectorQuantizerEMA(num_embed, embed_dim, beta, decay) if decay > 0.0 else VectorQuantizer(num_embed, embed_dim, beta)

        self.decoder = nn.Sequential(
            ResidualStack(embed_dim, num_residual),

            # nn.BatchNorm2d(embed_dim),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(embed_dim, embed_dim, (4, 4), (1, 2), 1, bias=False),

            nn.BatchNorm2d(embed_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(embed_dim, embed_dim, 4, 2, 1, bias=False),

            nn.BatchNorm2d(embed_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(embed_dim, embed_dim, 4, 2, 1, bias=False),

            nn.BatchNorm2d(embed_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(embed_dim, inc, 4, 2, 1),
        )

    def forward(self, x):
        z = self.encoder(x)
        q_loss, quantized, perplexity, _ = self.codebook(z)
        x_recon = self.decoder(quantized)

        return q_loss, x_recon, perplexity

    def get_codebooks_idx(self, x):
        z = self.encoder(x)
        _, _, _, codebooks_idx = self.codebook(z)

        return codebooks_idx


if __name__ == "__main__":
    model = VectorQuantizedVAE(inc=1)
    loss, x_recon, perplexity = model(torch.rand((2, 1, 88, 1024)))

    print(x_recon.shape)
