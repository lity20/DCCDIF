import os
from datetime import datetime
import numpy as np
from torch.utils.data import Dataset


class ShapeNet(Dataset):
    def __init__(self, config):
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Building ShapeNet dataset.')
        super().__init__()
        self.config = config

        with open(os.path.join(config['data_root'], config['split']+'_split.txt'), 'r') as f:
            split = f.readlines()
        self.split = [l.rstrip() for l in split]

        if config['load_in_memory']:
            self.data = []
            for i, data_id in enumerate(self.split):
                print('loading data: %d/%d...' % (i, len(self.split)), end='\r')
                data_path = os.path.join(config['data_root'], data_id+'.npy')
                self.data.append(np.load(data_path))

        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Dataset done.')

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        if self.config['load_in_memory']:
            xyz_sd = self.data[idx]
        else:
            data_path = os.path.join(self.config['data_root'], self.split[idx]+'.npy')
            xyz_sd = np.load(data_path)
        n_total_samples = xyz_sd.shape[0]
        xyz_sd = xyz_sd[np.random.randint(0, n_total_samples, size=self.config['num_samples_per_step'])]
        return xyz_sd[:, :3], xyz_sd[:, -1], idx
