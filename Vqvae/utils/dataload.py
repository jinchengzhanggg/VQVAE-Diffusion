import sys
import numpy as np
import pandas as pd
from utils.midi_transfer import midi2pianoroll
import os
from torch.utils.data import DataLoader, Dataset
import datetime
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def print_logger(dir_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    handler = logging.FileHandler("{}/{}.log".format(dir_path, name))
    console = logging.StreamHandler()

    logger.addHandler(handler)
    logger.addHandler(console)

    logger.info("Start print log")
    return logger


def read_path(dir_path, test_size=0.12):
    df = pd.read_csv(dir_path + "/maestro-v3.0.0.csv", header=0)
    df = df[df['canonical_composer'].isin(['Frédéric Chopin', 'Franz Liszt', 'Franz Schubert'])]
    ls = []
    for i in df['canonical_title'].unique():
        ls.append(df[df['canonical_title'] == i].iloc[0])
    subset = pd.DataFrame(ls)
    subset['composer_id'] = subset['canonical_composer'].astype("category").cat.codes

    # subset.to_csv("data_diffusion.csv", index=False)

    dt_trn = subset[subset['split'].isin(['train'])]
    dt_val = subset[subset['split'].isin(['test'])]

    trn_name_label = list(zip(dt_trn['midi_filename'].to_numpy(), dt_trn['composer_id'].to_numpy()))
    val_name_label = list(zip(dt_val['midi_filename'].to_numpy(), dt_val['composer_id'].to_numpy()))


    trn_name_label, val_name_label = train_test_split(trn_name_label + val_name_label,
                                                      test_size=test_size)
    print('train_midi_num: {}  valid_midi_num: {}'.format(len(trn_name_label), len(val_name_label)))

    return trn_name_label, val_name_label


def load_maestro_data(dir_path, test_size, window_size, stride):
    trn_name_label, val_name_label = read_path(dir_path, test_size)

    # trn_name_label = trn_name_label[:5]
    # val_name_label = val_name_label[:5]

    # training
    trn_data, trn_label = [], []
    for tune, label in tqdm(trn_name_label):
        quarters = midi2pianoroll(os.path.join(dir_path, tune), window_size=window_size, stride=stride)
        trn_data.extend(quarters)
        trn_label.extend(len(quarters)*[label])

    # validation
    val_data, val_label = [], []
    for tune, label in tqdm(val_name_label):
        quarters = midi2pianoroll(os.path.join(dir_path, tune), window_size=window_size, stride=stride)
        val_data.extend(quarters)
        val_label.extend(len(quarters)*[label])

    trn_data, val_data = np.array(trn_data, dtype=np.float32), np.array(val_data, dtype=np.float32)
    print('train data shape: {}'.format(trn_data.shape))
    print('valid data shape: {}'.format(val_data.shape))

    return list(zip(trn_data, trn_label)), list(zip(val_data, val_label)), (os.path.join(dir_path, trn_name_label[0][0]), os.path.join(dir_path, val_name_label[0][0]))


def load_maestro_data2(dir_path, test_size, window_size, stride):
    trn_name_label, val_name_label = read_path(dir_path, test_size)

    # trn_name_label = trn_name_label[:5]
    # val_name_label = val_name_label[:5]

    # training
    trn_data, trn_label, trn_path = [], [], []
    for tune, label in tqdm(trn_name_label):
        quarters = midi2pianoroll(os.path.join(dir_path, tune), window_size=window_size, stride=stride)
        trn_data.extend(quarters)
        trn_label.extend(len(quarters)*[label])
        trn_path.extend(len(quarters)*[os.path.join(dir_path, tune)])

    # validation
    val_data, val_label, val_path = [], [], []
    for tune, label in tqdm(val_name_label):
        quarters = midi2pianoroll(os.path.join(dir_path, tune), window_size=window_size, stride=stride)
        val_data.extend(quarters)
        val_label.extend(len(quarters)*[label])
        val_path.extend(len(quarters)*[os.path.join(dir_path, tune)])

    trn_data, val_data = np.array(trn_data, dtype=np.float32), np.array(val_data, dtype=np.float32)
    print('train data shape: {}'.format(trn_data.shape))
    print('valid data shape: {}'.format(val_data.shape))

    return list(zip(trn_data, trn_label, trn_path)), list(zip(val_data, val_label, val_path))



def load_midi_data(dir_path, window_size, stride):
    from glob import glob

    midi_path = glob(dir_path + '/*.mid')
    trn_name_label, val_name_label = midi_path[:-1], midi_path[-1:]

    # training
    trn_data, trn_label = [], []
    for tune in tqdm(trn_name_label):
        quarters = midi2pianoroll(tune, window_size=window_size, stride=stride)
        trn_data.extend(quarters)
        trn_label.extend(len(quarters) * [0])

    # validation
    val_data, val_label = [], []
    for tune in tqdm(val_name_label[:1]):
        quarters = midi2pianoroll(tune, window_size=window_size, stride=stride)
        val_data.extend(quarters)
        val_label.extend(len(quarters) * [0])

    trn_data, val_data = np.array(trn_data, dtype=np.float32), np.array(val_data, dtype=np.float32)
    print('train data shape: {}'.format(trn_data.shape))
    print('valid data shape: {}'.format(val_data.shape))

    return list(zip(trn_data, trn_label)), list(zip(val_data, val_label)), (trn_name_label[1], val_name_label[0])


class MidiDatasetVqvae(Dataset):
    def __init__(self, quarters, shape):
        data, label = list(zip(*quarters))
        self.data = np.stack(data, axis=0).reshape(-1, *shape)
        self.label = np.stack(label, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

class MidiDatasetVqvae2(Dataset):
    def __init__(self, quarters, shape):
        data, label, paths = list(zip(*quarters))
        self.data = np.stack(data, axis=0).reshape(-1, *shape)
        self.label = np.stack(label, axis=0)
        self.paths = paths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.paths[idx]


if __name__ == "__main__":
    dir_path = "../../datasets/midi"
    trn_data, val_data = load_midi_data(dir_path, window_size=4096, stride=1000)

    iterTrain = DataLoader(MidiDatasetVqvae(trn_data), batch_size=2)
    for x in iterTrain:
        print(x.shape)


