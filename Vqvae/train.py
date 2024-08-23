import os
import numpy as np
import torch
from utils.dataload import MidiDatasetVqvae, load_maestro_data, print_logger, load_midi_data
from torch.utils.data import DataLoader
from model.vqvae import VectorQuantizedVAE
from utils.midi_transfer import midi2pianoroll, pianoroll2midi
from types import SimpleNamespace
import yaml
from tqdm import tqdm


class Lightning:
    def __init__(self, config_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with open(config_path,  'r') as fid:
            cfg = yaml.safe_load(fid)
            self.args = SimpleNamespace(**cfg)

        self.save_path = 'runs/checkpoints-{}'.format(self.args.model_name)
        os.makedirs(self.save_path, exist_ok=True)

        self.logs = print_logger(self.save_path)
        self.generator_data()
        self.build_model()
        self.logs.info(cfg)

    def generator_data(self):
        trn_data, val_data, self.vis_data = eval(self.args.data_func['func'])(**self.args.data_func['params'])
        self.iterTrain = DataLoader(MidiDatasetVqvae(trn_data, self.args.shape), batch_size=self.args.batch_size, shuffle=True, pin_memory=True)
        self.iterValid = DataLoader(MidiDatasetVqvae(val_data, self.args.shape), batch_size=self.args.batch_size, pin_memory=True)

    def build_model(self):
        self.model = VectorQuantizedVAE(**self.args.net_params).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, amsgrad=False)

    def compute_accr(self, x, x_recon):
        x = x.detach().cpu().numpy()
        x_recon = x_recon.sigmoid().detach().round().cpu().numpy()

        equal = x == x_recon
        TP = equal[x==1].mean()
        FP = equal[x==0].mean()

        return TP, FP

    def compute_loss(self, x_recon, x):
        loss = self.criterion(x_recon, x)
        weight = ((x == 1) * 9 + 1).float()
        loss = (weight*loss).mean()

        # loss = self.criterion(x_recon, x).mean()

        return loss

    def train(self):
        self.model.train()

        loss, accr_TP, accr_FP = [], [], []
        for batch in tqdm(self.iterTrain):
            x = batch.to(self.device)
            q_loss, x_recon, perplexity = self.model(x)
            recon_loss = self.compute_loss(x_recon, x)
            _loss = recon_loss + q_loss

            self.optimizer.zero_grad()
            _loss.backward()
            self.optimizer.step()

            _accr_TP, _accr_FP = self.compute_accr(x, x_recon)
            loss.append(recon_loss.item())
            accr_TP.append(_accr_TP)
            accr_FP.append(_accr_FP)

        return np.mean(loss), np.mean(accr_TP), np.mean(accr_FP)

    def valid(self):
        self.model.eval()

        with torch.no_grad():
            loss, accr_TP, accr_FP = [], [], []
            for batch in tqdm(self.iterValid):
                x = batch.to(self.device)
                _, x_recon, _ = self.model(x)
                recon_loss = self.compute_loss(x_recon, x)


                _accr_TP, _accr_FP = self.compute_accr(x, x_recon)
                loss.append(recon_loss.item())
                accr_TP.append(_accr_TP)
                accr_FP.append(_accr_FP)

        return np.mean(loss), np.mean(accr_TP), np.mean(accr_FP)

    def generate(self, epoch):
        self.model.eval()


        for name, path in zip(['train', 'valid'], self.vis_data):
            x = midi2pianoroll(path, window_size=self.args.data_func['params']['window_size'], stride=self.args.data_func['params']['stride'])

            inputs = x.reshape(-1, *self.args.shape).astype(np.float32)
            with torch.no_grad():
                _, y, _ = self.model(torch.from_numpy(inputs[:1]).to(self.device))
            y = y.sigmoid().detach().round().cpu().numpy().squeeze()

            # save
            save_path = self.save_path + '/recon'
            os.makedirs(save_path, exist_ok=True)

            midi = pianoroll2midi(x[0])
            midi.write('{}/{}-ori.midi'.format(save_path, name, os.path.basename(path)[:-5]))

            midi = pianoroll2midi(y.reshape(self.args.shape[0], -1)) if self.args.shape[0] > 1 else pianoroll2midi(y)
            midi.write('{}/{}-epoch_{:05d}-{}-recon.midi'.format(save_path, name, epoch, os.path.basename(path)[:-5]))

    def fit(self):
        best_accr1, best_accr2 = 1e10, 1e10
        for epoch in range(self.args.epochs):
            loss_trn, accr_trn_TP, accr_trn_FP = self.train()
            loss_val, accr_val_TP, accr_val_FP = self.valid()

            self.logs.info('epoch[{:02d}]'.format(epoch))
            self.logs.info('train_rec_loss: {:.6f} accr_trn_TP: {:.6f} accr_trn_FP: {:.6f}'.format(loss_trn, accr_trn_TP, accr_trn_FP))
            self.logs.info('valid_rec_loss: {:.6f} accr_val_TP: {:.6f} accr_val_FP: {:.6f}'.format(loss_val, accr_val_TP, accr_val_FP))

            # save
            if accr_trn_TP > best_accr1:
                best_accr1 = accr_trn_TP
                torch.save(self.model.state_dict(), '{}/{}_best_train.pt'.format(self.save_path, self.args.model_name))

            if accr_val_TP > best_accr2:
                best_accr2 = accr_val_TP
                torch.save(self.model.state_dict(), '{}/{}_best_valid.pt'.format(self.save_path, self.args.model_name))

            if (epoch+1) % self.args.vis_epoch == 0:
                self.generate(epoch)
                torch.save(self.model.state_dict(), '{}/{}_{}.pt'.format(self.save_path, self.args.model_name, epoch))


if  __name__ == "__main__":
    light = Lightning('cfg_maestro_conv2d.yml')
    light.fit()