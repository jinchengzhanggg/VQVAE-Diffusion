import torch
import torch.nn as nn
import numpy as np
import yaml

import sys
sys.path.append('../Vqvae')
from check import load_model_vqvae


class Taming_vqvae(nn.Module):
    def __init__(self, trainable, params, ckpt_path, spatial_size):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with open(params) as fid:
            cfg_vqvae = yaml.safe_load(fid)

        vqvae = load_model_vqvae(cfg_vqvae['net_params'], ckpt_path, self.device)

        self.num_embeddings = vqvae.codebook._num_embeddings
        self.embedding = vqvae.codebook._embedding
        self.emd_num = vqvae.codebook._embedding_dim
        self.spatial_size = spatial_size

        self.dec = vqvae.decoder
        self.trainable = trainable
        self._set_trainable()

    def get_tokens(self, imgs, **kwargs):
        output = {'token': imgs}
        # output = {'token': rearrange(code, 'b h w -> b (h w)')}
        return output

    def _set_trainable(self):
        if not self.trainable:
            for pn, p in self.named_parameters():
                p.requires_grad = False
            self.eval()

    def decode(self, img_seq):
        img_seq = img_seq[0]
        

        encoding_indices = img_seq.view(-1, 1)  # torch.randint(0, 17, (16,1), device=device)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=self.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(torch.Size([1, *self.spatial_size, self.emd_num]))
        quantized = quantized.detach().permute(0, 3, 1, 2).contiguous()
        generated = np.round(self.dec(quantized).sigmoid().cpu().detach().numpy().squeeze())

        return img_seq, generated