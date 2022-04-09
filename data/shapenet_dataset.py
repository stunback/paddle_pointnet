import os
import sys
import numpy as np
import paddle
import h5py
import tqdm

sys.path.append('../')
from utils import visualization


class ShapeNetDataset(paddle.io.Dataset):
    """
    ShapeNet Dataset
    """
    def __init__(self, root_path: str, mode: str = 'train'):
        """
        :param root_path: str, root path of shapenet dataset.
        :param mode: str, 'train', 'val' or 'test'
        """
        super(ShapeNetDataset, self).__init__()
        self.data_path = os.path.join(root_path, 'hdf5_data')
        assert mode in ['train', 'val', 'test'], 'Unsupported mode: {}'.format(mode)
        self.mode = mode
        train_file_list_path = os.path.join(self.data_path, 'train_hdf5_file_list.txt')
        val_file_list_path = os.path.join(self.data_path, 'val_hdf5_file_list.txt')
        test_file_list_path = os.path.join(self.data_path, 'test_hdf5_file_list.txt')
        self.train_list = self.generate_file_list(train_file_list_path)
        self.val_list = self.generate_file_list(val_file_list_path)
        self.test_list = self.generate_file_list(test_file_list_path)
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
        elif self.mode == 'val':
            file_list = self.val_list
        else:
            file_list = self.test_list
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
        data = paddle.to_tensor(self.datas[index].transpose((1, 0)), dtype=paddle.float32)
        label = paddle.to_tensor(self.labels[index], dtype=paddle.int32)

        return data, label

    def __len__(self):
        return len(self.labels)


def get_dataloader(dataset_path: str, batch_size: int = 16, num_workers: int = 0, if_test: bool = False):
    """
    :param dataset_path: str
    :param batch_size: int
    :param num_workers: int
    :param if_test: bool
    :return: (paddle.io.Dataloader, paddle.io.Dataloader) for train, val
             or paddle.io.Dataloader for test
    """
    if not if_test :
        train_dataset = ShapeNetDataset(dataset_path, 'train')
        val_dataset = ShapeNetDataset(dataset_path, 'val')

        train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
        val_dataset = paddle.io.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
        return train_loader, val_dataset
    else:
        test_dataset = ShapeNetDataset(dataset_path, 'test')
        test_dataset = paddle.io.DataLoader(test_dataset, batch_size=batch_size)
        return test_dataset


if __name__ == '__main__':
    path = 'E:/ShapeNetDataset'
    ShapeNetDataset(path)
    '''
    path = 'E:/ShapeNetDataset/hdf5_data'
    assert os.path.exists(path), 'The dataset path {} does not exist'.format(path)

    data_path = os.path.join(path, 'ply_data_train5.h5')
    data = h5py.File(data_path)
    pcds = data['data']
    labels = data['label']
    pids = data['pid']
    print(labels, labels[0], pids, pids[0])
    labels_ = np.array(labels)
    print(labels_)
    # for pcd in pcds:
    #    visualization.vis_pcd(pcd)
    '''
