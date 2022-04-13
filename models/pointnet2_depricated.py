import sys
import numpy as np
import paddle

sys.path.append('./')
from models.base import Conv1DBN, Conv1DBlock
from models.pointnet import PointNet


class PointNetSAModuleSSG(paddle.nn.Layer):
    """
    PointNet2 structure
    """

    def __init__(self, M, r, k, in_channels, mlp, group_all=False, pooling='max', use_pcd=True):
        super(PointNetSAModuleSSG, self).__init__()
        self.M = M
        self.radius = r
        self.k = k
        self.in_channels = in_channels
        self.mlp = mlp
        self.group_all = group_all
        self.pooling = pooling
        self.use_pcd = use_pcd
        self.backbone = paddle.nn.Sequential()
        for i, out_channels in enumerate(mlp):
            self.backbone.add_sublayer('Conv{}'.format(i), paddle.nn.Conv2D(in_channels, out_channels, 1, stride=1, padding=0))
            self.backbone.add_sublayer('Bn{}'.format(i), paddle.nn.BatchNorm2D(out_channels))
            self.backbone.add_sublayer('Relu{}'.format(i), paddle.nn.ReLU())
            in_channels = out_channels

    def forward(self, pcd, points=None):
        if self.group_all:
            new_pcd, new_points, grouped_inds, grouped_pcd = self.sample_and_group_all(pcd, points, self.use_pcd)
        else:
            new_pcd, new_points, grouped_inds, grouped_pcd = self.sample_and_group(pcd=pcd,
                                                                                   points=points,
                                                                                   M=self.M,
                                                                                   radius=self.radius,
                                                                                   k=self.k,
                                                                                   use_pcd=self.use_pcd)
        new_points = self.backbone(new_points.transpose([0, 3, 2, 1]).contiguous())
        if self.pooling == 'avg':
            new_points = paddle.mean(new_points, axis=2)
        else:
            new_points = paddle.max(new_points, axis=2)[0]
        new_points = new_points.transpose([0, 2, 1]).contiguous()

        return new_pcd, new_points

    @staticmethod
    def get_pcd_distance(points1: paddle.Tensor, points2: paddle.Tensor):
        """
        Calculate dists between two group points
        :param points1: shape=(B, M, C)
        :param points2: shape=(B, N, C)
        :return: paddle.Tensor
        """
        B, N, _ = points1.shape
        _, M, _ = points2.shape
        dist = -2 * paddle.matmul(points1.cast(paddle.float32), points2.transpose([0, 2, 1]).cast(paddle.float32))
        dist += paddle.sum(points1 ** 2, axis=-1).reshape([B, N, 1])
        dist += paddle.sum(points2 ** 2, axis=-1).reshape([B, 1, M])

        return dist

    @staticmethod
    def gather_points(points, inds):
        """
        :param points: shape=(B, N, C)
        :param inds: shape=(B, M) or shape=(B, M, K)
        :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
        """
        B, N, C = points.shape
        inds_shape = list(inds.shape)
        inds_shape[1:] = [1] * len(inds_shape[1:])
        repeat_shape = list(inds.shape)
        repeat_shape[0] = 1
        batchlists = paddle.arange(0, B, dtype=paddle.int64).reshape(inds_shape).tile(repeat_shape)
        print(points, batchlists, inds)

        return paddle.to_tensor(points.numpy()[batchlists.numpy(), inds.numpy(), :])

    def farthest_pcd_sample(self, pcd: paddle.Tensor, npoint: int):
        """
        Sampling npoint farthest points in a point cloud
        :param pcd: paddle.Tensor, shape=[B, N, 3]
        :param npoint: int
        :return: paddle.Tensor, shape=[B, npoint]
        """
        B, N, C = pcd.shape
        centroids = paddle.zeros(shape=[B, npoint], dtype=paddle.int64)
        dists = paddle.ones([B, N]) * 1e10
        inds = paddle.randint(0, N, shape=[B], dtype=paddle.int64)
        batchlists = paddle.arange(0, B, dtype=paddle.int64)

        for i in range(npoint):
            centroids[:, i] = inds.cast(paddle.int64)
            cur_point = paddle.to_tensor(pcd.numpy()[batchlists.numpy().astype(np.int32), inds.numpy().astype(np.int32), :], dtype=paddle.int64)  # (B, 3)
            cur_dist = paddle.squeeze(self.get_pcd_distance(paddle.unsqueeze(cur_point, 1), pcd), axis=1)
            dists[cur_dist < dists] = cur_dist[cur_dist < dists]
            inds = paddle.max(dists, axis=1)[1]

        return centroids

    def query_ball_point(self, pcd: paddle.Tensor, pcd_new: paddle.Tensor, radius: float, k: int):
        """
        get npoint groups
        :param pcd: paddle.Tensor, shape=[B, N, 3]
        :param pcd_new: paddle.Tensor, shape=[B, npoint, 3]
        :param radius: float
        :param k: int, limit the points repeat
        :return: paddle.Tensor, shape=[B, M, k]
        """
        B, N, C = pcd.shape
        M = pcd_new.shape[1]
        grouped_inds = paddle.arange(0, N, dtype=paddle.int64).reshape([1, 1, N]).tile([B, M, 1])
        dists = self.get_pcd_distance(pcd_new, pcd)
        grouped_inds[dists > radius] = N
        grouped_inds = paddle.sort(grouped_inds, axis=-1)[0][:, :, :k]
        grouped_min_inds = grouped_inds[:, :, 0:1].tile([1, 1, k])
        grouped_inds[grouped_inds == N] = grouped_min_inds[grouped_inds == N]

        return grouped_inds

    def sample_and_group(self, pcd, points, M, radius, k, use_pcd=True):
        """
        :param pcd: shape=(B, N, 3)
        :param points: shape=(B, N, C)
        :param M: int
        :param radius:float
        :param k: int
        :param use_pcd: bool, if True concat pcd with local point features, otherwise just use point features
        :return: new_pcd, shape=(B, M, 3); new_points, shape=(B, M, K, C+3);
                 group_inds, shape=(B, M, K); grouped_pcd, shape=(B, M, K, 3)
        """
        new_pcd = self.gather_points(pcd, self.farthest_pcd_sample(pcd, M))
        grouped_inds = self.query_ball_point(pcd, new_pcd, radius, k)
        grouped_pcd = self.gather_points(pcd, grouped_inds)
        grouped_pcd -= paddle.unsqueeze(new_pcd, 2).tile(1, 1, k, 1)
        if points is not None:
            grouped_points = self.gather_points(points, grouped_inds)
            if use_pcd:
                new_points = paddle.concat([grouped_pcd.float(), grouped_points.float()], axis=-1)
            else:
                new_points = grouped_points
        else:
            new_points = grouped_pcd

        return new_pcd, new_points, grouped_inds, grouped_pcd

    @staticmethod
    def sample_and_group_all(pcd, points, use_pcd=True):
        """
        :param pcd: shape=(B, M, 3)
        :param points: shape=(B, M, C)
        :param use_pcd:
        :return: new_pcd, shape=(B, 1, 3); new_points, shape=(B, 1, M, C+3);
                 group_inds, shape=(B, 1, M); grouped_pcd, shape=(B, 1, M, 3)
        """
        B, M, C = pcd.shape
        new_pcd = paddle.zeros([B, 1, C])
        grouped_inds = paddle.arange(0, M, dtype=paddle.int64).reshape([1, 1, M]).tile([B, 1, 1])
        grouped_pcd = pcd.reshape([B, 1, M, C])
        if points is not None:
            if use_pcd:
                new_points = paddle.concat([pcd.float(), points.float()], axis=2)
            else:
                new_points = points
            new_points = paddle.unsqueeze(new_points, axis=1)
        else:
            new_points = grouped_pcd
        return new_pcd, new_points, grouped_inds, grouped_pcd


if __name__ == '__main__':
    M, radius, K = 5, 0.2, 6
    net = PointNetSAModuleSSG(M, radius, K, 6, [32, 64, 128])
    paddle.summary(net, (4, 3, 1024))
