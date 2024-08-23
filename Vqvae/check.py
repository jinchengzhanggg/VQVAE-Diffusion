import os.path
from glob import glob
from utils.midi_transfer import midi2pianoroll, pianoroll2midi
from model.vqvae import VectorQuantizedVAE
import torch
import pickle
import numpy as np
from omegaconf import OmegaConf as ome


def load_model_vqvae(params, ckpt_path, device):
    model = VectorQuantizedVAE(**params).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    return model


def decode(img_seq, vqvae, spatial_size=[11, 128], device='cuda'):
    num_embeddings = vqvae.codebook._num_embeddings
    embedding = vqvae.codebook._embedding
    emd_num = vqvae.codebook._embedding_dim
    dec = vqvae.decoder

    encoding_indices = img_seq.view(-1, 1)  # torch.randint(0, 17, (16,1), device=device)
    encodings = torch.zeros(encoding_indices.shape[0], num_embeddings, device=device)
    encodings.scatter_(1, encoding_indices, 1)

    # Quantize and unflatten
    quantized = torch.matmul(encodings, embedding.weight).view(torch.Size([1, *spatial_size, emd_num]))
    quantized = quantized.detach().permute(0, 3, 1, 2).contiguous()
    generated = np.round(dec(quantized).sigmoid().cpu().detach().numpy().squeeze())

    return generated


def run(recon_dir, cfg_path, ckpt_path, pkl_path):
    os.makedirs(recon_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = ome.load(cfg_path)
    vqvae = load_model_vqvae(cfg['net_params'], ckpt_path, device=device)

    with open(pkl_path, 'rb') as fid:
        data = pickle.load(fid)
    
    dt = data['trn_data']
    paths = data['trn_paths']

    for i in range(10):
        ph = paths[i]
        x = torch.from_numpy(dt[0][i]).long().to(device)
        y = decode(x, vqvae, device=device)

        midi = pianoroll2midi(y)

        name = '-'.join(ph.split('/')[-2:])
        midi.write(f'{recon_dir}/{i:04d}-{name}')



if __name__ == '__main__':
    recon_dir = 'check_midi'
    cfg_path = 'cfg_maestro_conv2d.yml'
    ckpt_path = 'runs/checkpoints-exp-20230720/exp-20230720_72.pt'
    pkl_path = '../datasets/vqvae_encode_20230720.pkl'

    run(recon_dir, cfg_path, ckpt_path, pkl_path)

    
