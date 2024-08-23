from torch.utils.data import Dataset
import pickle


class MasetroDataset(Dataset):
    def __init__(self, pkl_path, is_train, data_root=''):
        super().__init__()
        with open(pkl_path, 'rb') as fid:
            data = pickle.load(fid)

        self.dataX, self.dataY = data['trn_data'] if is_train else data['val_data']
        print('data: ', self.dataX.shape)

    def __len__(self):
        return len(self.dataX)

    def __getitem__(self, idx):
        x = self.dataX[idx]
        y = self.dataY[idx]

        return {'image': x, 'label': y}

