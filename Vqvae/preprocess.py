import numpy as np
import os
from tqdm import tqdm
import pickle
import yaml
from utils.dataload import midi2pianoroll, read_path, MidiDatasetVqvae2, load_maestro_data2
from torch.utils.data import DataLoader
import torch
from model.vqvae import VectorQuantizedVAE
from omegaconf import OmegaConf as ome


def midi2idx(vqvae, iterData, device):
    data, label, paths = [], [], []
    with torch.no_grad():
        for x, y, ph in tqdm(iterData):
            idx = vqvae.get_codebooks_idx(x.to(device))
            data.append(idx.cpu().numpy().reshape(idx.shape[0], -1))
            label.append(y.numpy())
            paths.extend(ph)
    dataX, dataY = np.concatenate(data, axis=0), np.concatenate(label)

    return (dataX, dataY), paths


def load_model(ckpt_path, cfg, device):
    model = VectorQuantizedVAE(**cfg['net_params']).to(device)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    return model


def run_midi2vetcor(config_path, ckpt_path, save_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    cfg = ome.load(config_path)


    vqvae = load_model(ckpt_path, cfg, device)


    trn_data, val_data = load_maestro_data2(**cfg.data_func['params'])


    iterTrain = DataLoader(MidiDatasetVqvae2(trn_data, cfg['shape']), batch_size=cfg['batch_size'])
    itervalid = DataLoader(MidiDatasetVqvae2(val_data, cfg['shape']), batch_size=cfg['batch_size'])

    print("train midi to index ...")
    trn_data, trn_paths = midi2idx(vqvae, iterTrain, device)
    val_data, val_paths = midi2idx(vqvae, itervalid, device)
    print("train data: {}\nvalid data: {}".format(trn_data[0].shape, val_data[0].shape))

    # save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as fid:
        pickle.dump({'trn_data': trn_data, 'val_data': val_data, 'trn_paths': trn_paths, 'val_paths': val_paths}, fid)


if __name__ == '__main__':
    run_midi2vetcor('cfg_maestro_conv2d.yml', 
                    'runs/checkpoints-exp-20230720/exp-20230720_72.pt',
                    '../datasets/vqvae_encode_20230720.pkl')