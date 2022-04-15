import numpy as np
import paddle
import sys

sys.path.append('./')
from models.base import Conv1DBN, Conv1DBlock


def knn(x: paddle.Tensor, k: int):
    """
    point cloud k nearest neighbor
    :param x: paddle.Tensor, shape=[b, 3, n]
    :param k: int
    :return: paddle.Tensor, shape=[b, n, k]
    """
    inner = -2 * paddle.matmul(x.transpose([0, 2, 1]), x)
    xx = paddle.sum(x ** 2, axis=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose([0, 2, 1])
    idx = paddle.topk(pairwise_distance, k=k, axis=-1)[1]

    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Edge Conv
    :param x: paddle.Tensor, shape=[b, 3, n]
    :param k: int
    :param idx: paddle.Tensor, shape=[b, k], optional
    :return: paddle.Tensor, shape=[b, 6, n, k]
    """
    batch_size, channels, num_points = x.shape
    if idx is None:
        idx = knn(x, k=k)

    idx_base = paddle.arange(0, batch_size).reshape([-1, 1, 1]) * num_points
    idx = (idx + idx_base).reshape([-1])

    x = paddle.transpose(x, [0, 2, 1])
    feature = x.reshape([batch_size * num_points, -1])[idx]
    feature = feature.reshape([batch_size, num_points, k, channels])
    x = x.reshape([batch_size, num_points, 1, channels]).tile([1, 1, k, 1])
    feature = paddle.concat([feature - x, x], axis=3).transpose([0, 3, 1, 2])

    return feature


class DGCNN(paddle.nn.Layer):
    """
    DGCNN structure
    """

    def __init__(self, in_channels=3, num_points=1024, k=20, num_features=1024, num_classes=40):
        super(DGCNN, self).__init__()
        self.in_channels = in_channels
        self.num_points = num_points
        self.k = k
        self.num_features = num_features
        self.num_classes = num_classes

        self.conv1 = paddle.nn.Sequential(paddle.nn.Conv2D(6, 64, kernel_size=1),
                                          paddle.nn.BatchNorm2D(64),
                                          paddle.nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = paddle.nn.Sequential(paddle.nn.Conv2D(64 * 2, 64, kernel_size=1),
                                          paddle.nn.BatchNorm2D(64),
                                          paddle.nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = paddle.nn.Sequential(paddle.nn.Conv2D(64 * 2, 128, kernel_size=1),
                                          paddle.nn.BatchNorm2D(128),
                                          paddle.nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = paddle.nn.Sequential(paddle.nn.Conv2D(128 * 2, 256, kernel_size=1),
                                          paddle.nn.BatchNorm2D(256),
                                          paddle.nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = paddle.nn.Sequential(paddle.nn.Conv1D(512, self.num_features, kernel_size=1),
                                          paddle.nn.BatchNorm1D(self.num_features),
                                          paddle.nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = paddle.nn.Linear(self.num_features * 2, 512)
        self.bn6 = paddle.nn.BatchNorm1D(512)
        self.dp1 = paddle.nn.Dropout(p=0.5)
        self.linear2 = paddle.nn.Linear(512, 256)
        self.bn7 = paddle.nn.BatchNorm1D(256)
        self.dp2 = paddle.nn.Dropout(p=0.5)
        self.linear3 = paddle.nn.Linear(256, self.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(axis=-1, keepdim=False)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(axis=-1, keepdim=False)

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(axis=-1, keepdim=False)

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(axis=-1, keepdim=False)

        x = paddle.concat([x1, x2, x3, x4], axis=1)
        x = self.conv5(x)
        x1 = paddle.nn.functional.adaptive_max_pool1d(x, 1).reshape([batch_size, -1])
        x2 = paddle.nn.functional.adaptive_avg_pool1d(x, 1).reshape([batch_size, -1])
        x = paddle.concat([x1, x2], axis=1)

        x = paddle.nn.functional.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = paddle.nn.functional.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x


if __name__ == '__main__':
    a = paddle.to_tensor([[[-1, -1, -1],
                           [0, 0, 0],
                           [-2, -2, -2],
                           [-1.5, -1.5, -1.5],
                           [-2.2, -2.2, -2.2]],
                          [[0, 0, 0],
                           [0.5, 0.5, 0.2],
                           [0.2, 0.2, 0.2],
                           [0.7, 0.7, 0.7],
                           [1, 1, 1]]]).transpose([0, 2, 1])
    # i = knn(a, 2)
    # print(i)
    f = get_graph_feature(a, 2)
    print(f)
    i = paddle.to_tensor([1, 2], dtype=paddle.int64)
    ii = paddle.to_tensor([2, 4], dtype=paddle.int64)
    print(a[:])
    print(a[i-1, i][:])
    net = DGCNN()
    paddle.summary(net, (32, 3, 1024))
