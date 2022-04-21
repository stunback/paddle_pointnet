import sys
import numpy as np
import paddle

sys.path.append('./')
from models.base import Conv1DBN, Conv1DBlock
from models.pointnet import PointNet


def farthest_point_sampling(xyz: paddle.Tensor, nsamples: int) -> paddle.Tensor:
    B, N, C = xyz.shape
    idx = paddle.zeros([B, nsamples], dtype=paddle.int64)   # 用于存储最后选取点的索引.
    tmp = paddle.ones([B, N], dtype=paddle.float32) * 1e10   # 距离矩阵,以1e10作为初始化.
    farthest = paddle.randint(high=N, shape=[B])  # 用随机初始化的最远点
    batch_indices = paddle.arange(B)
    for i in range(nsamples):
        # 依次更新idx中的数据.共nsamples个
        idx[:, i] = farthest
        # 取出第i个点的xyz坐标
        centroids = xyz[batch_indices, farthest].reshape([B, 1, 3])
        # 计算距离
        dist = paddle.sum((xyz - centroids)**2, axis=-1)
        # 更新距离矩阵中数据,矩阵中距离在慢慢变小
        tmp = paddle.where(dist < tmp, dist, tmp)
        # 更新最远点
        farthest = paddle.argmax(tmp, axis=-1)

    return idx


def index_points(xyz: paddle.Tensor, idx: paddle.Tensor) -> paddle.Tensor:
    """
    按照索引idx 从xyz中取对应的点.
    :param xyz: 点云数据    [B, N, C]
    :param idx: 要选取的点的索引    [B, S]
    :return:    返回选取的点  [B, S, C]
    """
    B = xyz.shape[0]
    view_shape = idx.shape
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = idx.shape
    repeat_shape[0] = 1
    batch_indices = paddle.arange(B, dtype=paddle.int64).reshape(view_shape).tile(repeat_shape)
    new_points = xyz[batch_indices, idx]

    return new_points


def square_distance(src: paddle.Tensor, dst: paddle.Tensor) -> paddle.Tensor:
    """
    计算俩组坐标之间的欧氏距离,  (x-y)**2 = x**2 + y**2 - 2*xy
    :param src: [B, N, 3]
    :param dst: [B, S, 3]
    :return:    俩组坐标之间的俩俩对应距离的矩阵 size: [B, N, S]
    """
    B, N, C = src.shape
    S = dst.shape[1]
    dist = -2 * paddle.matmul(src, dst.transpose([0, 2, 1]))  # 2*(xn * xm + yn * ym + zn * zm)
    dist += paddle.sum(src ** 2, axis=-1).reshape([B, N, 1])  # xn*xn + yn*yn + zn*zn
    dist += paddle.sum(dst ** 2, axis=-1).reshape([B, 1, S])  # xm*xm + ym*ym + zm*zm

    return dist


def query_ball_point(xyz: paddle.Tensor, new_xyz: paddle.Tensor, radius: float, nsample: int) -> paddle.Tensor:
    """
    :param xyz: paddle.Tensor, shape=[B, N, C]
    :param new_xyz: paddle.Tensor, shape=[B, S, C]
    :param radius: float
    :param nsample: int
    return: paddle.Tensor, shape=[B, S, nsample]
    """
    B, N, C = xyz.shape
    S = new_xyz.shape[1]
    group_idx = paddle.arange(N).reshape([1, 1, N]).tile([B, S, 1])
    # xyz 与 xyz_new 之间坐标俩俩对应的距离矩阵
    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
    # 大于radius**2的,将group_idx 之间置为N.
    group_idx = paddle.where(sqrdists > radius ** 2, paddle.ones(shape=[B, S, N]) * N, group_idx.cast(paddle.float32))
    # 做升序排列，取出前nsample个点
    group_idx = paddle.sort(group_idx, axis=-1)[:, :, :nsample]
    # 对于数据不足的情况,直接将等于N的点替换为第一个点的值
    group_first = group_idx[:, :, 0].reshape([B, S, 1]).tile([1, 1, nsample])
    group_idx = paddle.where(group_idx == N, group_first, group_idx).cast(paddle.int64)

    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points=None, returnfps=False):
    """
    Input:
        npoint: int
        radius: float
        nsample: int
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sampling(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(xyz, new_xyz, radius, nsample)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.reshape([B, S, 1, C])

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = paddle.concat([grouped_xyz_norm, grouped_points], axis=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points=None):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    B, N, C = xyz.shape
    new_xyz = paddle.zeros([B, 1, C])
    grouped_xyz = xyz.reshape([B, 1, N, C])
    if points is not None:
        new_points = paddle.concat([grouped_xyz, points.reshape([B, 1, N, -1])], axis=-1)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points


class PointNetSetAbstraction(paddle.nn.Layer):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = paddle.nn.LayerList()
        self.mlp_bns = paddle.nn.LayerList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(paddle.nn.Conv2D(last_channel, out_channel, 1))
            self.mlp_bns.append(paddle.nn.BatchNorm2D(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz: paddle.Tensor, points: paddle.Tensor = None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.transpose([0, 2, 1])
        if points is not None:
            points = points.transpose([0, 2, 1])

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.transpose([0, 3, 2, 1])  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = paddle.nn.functional.relu(bn(conv(new_points)))

        new_points = paddle.max(new_points, axis=2)
        new_xyz = new_xyz.transpose([0, 2, 1])

        return new_xyz, new_points


class PointNetSetAbstractionMsg(paddle.nn.Layer):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        """
        :param npoint:      全局采样点
        :param radius_list: 局部采样半径
        :param nsample_list:局部采样点数
        :param in_channel:  输入通道数
        :param mlp_list:    输出通道数列表
        """
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = paddle.nn.LayerList()
        self.bn_blocks = paddle.nn.LayerList()
        for i in range(len(mlp_list)):
            convs = paddle.nn.LayerList()
            bns = paddle.nn.LayerList()
            last_channel = in_channel
            for out_channel in mlp_list[i]:
                convs.append(paddle.nn.Conv2D(last_channel, out_channel, 1))   # 特征提取使用1x1卷积实现. 可以理解为全连接.但是可以省却view操作.
                bns.append(paddle.nn.BatchNorm2D(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz: paddle.Tensor, points: paddle.Tensor = None):
        """
        :param xyz:         [B, 3 ,N]
        :param points:    [B, D, N]
        """
        xyz = xyz.transpose([0, 2, 1])   # [B, N, C]
        if points is not None:
            points = points.transpose([0, 2, 1])  # [B, N, D]

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sampling(xyz, S))     # [B, S, C] 提取全局采样点
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(xyz, new_xyz, radius, K)   # [B, S, K]
            grouped_xyz = index_points(xyz, group_idx)              # [B, S, K, C]  局部采样点
            grouped_xyz -= new_xyz.reshape([B, S, 1, C])                 # 这里可以理解为数据归一化
            if points is not None:
                grouped_points = index_points(points, group_idx)    # [B, N, k, D]  如有特征点,则对特征点进行对应的采样
                grouped_points = paddle.concat([grouped_points, grouped_xyz], axis=-1)   # [B, S, K, C+D]
            else:
                grouped_points = grouped_xyz    # [B, S, K, C]

            grouped_points = grouped_points.transpose([0, 3, 2, 1])  # [B, C, K, S] or [B, C+D, K, S]  K=nsample, S=npoint
            for j in range(len(self.conv_blocks[i])):   # 进行特征提取
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = paddle.nn.functional.relu(bn(conv(grouped_points)))   # [B, mlp_list[-1], K, S] or [B, mlp_list[-1], K, S]
            new_points = paddle.max(grouped_points, axis=2)    # [B, mlp_list[-1], S] 这里在第三个维度上取最大的. 这里可以理解为最大池化
            new_points_list.append(new_points)

        new_xyz = new_xyz.transpose([0, 2, 1])
        new_points_concat = paddle.concat(new_points_list, axis=1)   # [B, sum(mlp_list[-1]), S]

        return new_xyz, new_points_concat   # 在输出时, 需将提取的特征 以及对应的采样点xyz返回


class PointNet2SSG(paddle.nn.Layer):
    """
    PointNet++ SSG new
    """
    def __init__(self, num_classes=40, normal_channel=False):
        super(PointNet2SSG, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = paddle.nn.Linear(1024, 512)
        self.bn1 = paddle.nn.BatchNorm1D(512)
        self.drop1 = paddle.nn.Dropout(0.4)
        self.fc2 = paddle.nn.Linear(512, 256)
        self.bn2 = paddle.nn.BatchNorm1D(256)
        self.drop2 = paddle.nn.Dropout(0.4)
        self.fc3 = paddle.nn.Linear(256, num_classes)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.reshape([B, 1024])
        x = self.drop1(paddle.nn.functional.relu(self.bn1(self.fc1(x))))
        x = self.drop2(paddle.nn.functional.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x, l3_points


class PointNet2MSG(paddle.nn.Layer):
    def __init__(self, num_classes=40, normal_channel=False):
        super(PointNet2MSG, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320 + 3, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = paddle.nn.Linear(1024, 512)
        self.bn1 = paddle.nn.BatchNorm1D(512)
        self.drop1 = paddle.nn.Dropout(0.4)
        self.fc2 = paddle.nn.Linear(512, 256)
        self.bn2 = paddle.nn.BatchNorm1D(256)
        self.drop2 = paddle.nn.Dropout(0.5)
        self.fc3 = paddle.nn.Linear(256, num_classes)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.reshape([B, 1024])
        x = self.drop1(paddle.nn.functional.relu(self.bn1(self.fc1(x))))
        x = self.drop2(paddle.nn.functional.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x, l3_points


if __name__ == '__main__':
    # M, radius, K = 5, 0.2, 6
    # net = PointNetSAModuleSSG(M, radius, K, 6, [32, 64, 128])
    # paddle.summary(net, (4, 3, 1024))
    a = paddle.randn(shape=[2, 5, 3])
    f = farthest_point_sampling(a, 3)
    print(f)
    ff = index_points(a, f)
    print(a)
    print(ff)
    fff = query_ball_point(a, ff, 0.5, 2)
    print(fff)

    net = PointNet2SSG(40)
    paddle.summary(net, (32, 3, 1024))
