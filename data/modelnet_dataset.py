import os
import sys
import numpy as np
import h5py
import paddle
import tqdm

sys.path.append('../')
from utils import visualization
from data import augmentation


class ModelNetDataset(paddle.io.Dataset):
    """
    ModelNet Dataset
    """
    def __init__(self, root_path: str, mode: str = 'train'):
        """
        :param root_path: str
        :param mode: str, 'train' or 'val'
        """
        super(ModelNetDataset, self).__init__()
        self.data_path = os.path.join(root_path, 'modelnet40_ply_hdf5')
        assert mode in ['train', 'val'], 'Unsupported mode: {}'.format(mode)
        self.mode = mode
        train_file_list_path = os.path.join(self.data_path, 'train_files.txt')
        val_file_list_path = os.path.join(self.data_path, 'test_files.txt')
        self.train_list = self.generate_file_list(train_file_list_path)
        self.val_list = self.generate_file_list(val_file_list_path)
        self.datas, self.labels = self.generate_data()

    @staticmethod
    def generate_file_list(file_list_path: str):
        file_list = []
        f = open(file_list_path, 'r')
        files = f.readlines()
        for file in files:
            file_list.append(file.strip())
        f.close()

        return file_list

    def generate_data(self):
        datas = []
        labels = []
        if self.mode == 'train':
            file_list = self.train_list
        else:
            file_list = self.val_list

        for file in tqdm.tqdm(file_list, total=len(file_list)):
            file = os.path.join(self.data_path, file)
            f = h5py.File(file, 'r')
            data = f['data']
            label = f['label']
            datas.extend(data)
            labels.extend(label)
            f.close()

        return datas, labels

    def __getitem__(self, index):
        """
        :param index: int
        :return: (np.ndarry, np.ndarray), shape0=[2048, 3], shape1=[1]
        """
        if self.mode == 'train':
            datas_trans = self.datas[index]
            datas_trans = augmentation.augment_pcd(datas_trans, drop_rate=0., rotate_rad=0.)
        else:
            datas_trans = self.datas[index]
            datas_trans = augmentation.augment_pcd(datas_trans, None, None, None, None, None, True)
        data = paddle.to_tensor(datas_trans.transpose([1, 0]), dtype=paddle.float32)
        label = paddle.to_tensor(self.labels[index], dtype=paddle.int32)

        return data, label

    def __len__(self):
        return len(self.labels)


def get_dataloader(dataset_path: str, batch_size: int = 16, num_workers: int = 0):
    """
    :param dataset_path: str
    :param batch_size: int
    :param num_workers: int
    :return: (paddle.io.Dataloader, paddle.io.Dataloader) for train, val
             or paddle.io.Dataloader for test
    """
    train_dataset = ModelNetDataset(dataset_path, 'train')
    val_dataset = ModelNetDataset(dataset_path, 'val')

    train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_dataset = paddle.io.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_dataset


if __name__ == '__main__':
    path = 'E:/ShapeNetDataset'
    ModelNetDataset(path)

