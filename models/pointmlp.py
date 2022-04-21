import sys
import numpy as np
import paddle

sys.path.append('./')
from models.base import Conv1DBN, Conv1DBlock, Conv1DResBlock


def farthest_point_sampling(xyz: paddle.Tensor, nsamples: int) -> paddle.Tensor:
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        nsamples: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, C = xyz.shape
    idx = paddle.zeros([B, nsamples], dtype=paddle.int64)  # 用于存储最后选取点的索引.
    tmp = paddle.ones([B, N], dtype=paddle.float32) * 1e10  # 距离矩阵,以1e10作为初始化.
    farthest = paddle.randint(high=N, shape=[B])  # 用随机初始化的最远点
    batch_indices = paddle.arange(B)
    for i in range(nsamples):
        # 依次更新idx中的数据.共nsamples个
        idx[:, i] = farthest
        # 取出第i个点的xyz坐标
        centroids = xyz[batch_indices, farthest].reshape([B, 1, 3])
        # 计算距离
        dist = paddle.sum((xyz - centroids) ** 2, axis=-1)
        # 更新距离矩阵中数据,矩阵中距离在慢慢变小
        tmp = paddle.where(dist < tmp, dist, tmp)
        # 更新最远点
        farthest = paddle.argmax(tmp, axis=-1)

    return idx


def index_points(xyz: paddle.Tensor, idx: paddle.Tensor) -> paddle.Tensor:
    """
    Input:
        xyz: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
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
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, S, C]
    Output:
        dist: per-point square distance, [B, N, S]
    """
    B, N, C = src.shape
    S = dst.shape[1]
    dist = -2 * paddle.matmul(src, dst.transpose([0, 2, 1]))  # 2*(xn * xm + yn * ym + zn * zm)
    dist += paddle.sum(src ** 2, axis=-1).reshape([B, N, 1])  # xn*xn + yn*yn + zn*zn
    dist += paddle.sum(dst ** 2, axis=-1).reshape([B, 1, S])  # xm*xm + ym*ym + zm*zm

    return dist


def query_ball_point(xyz: paddle.Tensor, new_xyz: paddle.Tensor, radius: float, nsample: int) -> paddle.Tensor:
    """
    Input:
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
        radius: local region radius
        nsample: max sample number in local region
    Return:
        group_idx: grouped points index, [B, S, nsample]
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


def knn_point(xyz: paddle.Tensor, new_xyz: paddle.Tensor, nsample: int):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = paddle.topk(sqrdists, nsample, axis=-1, largest=False, sorted=False)

    return group_idx


class LocalGrouper(paddle.nn.Layer):
    def __init__(self, channel, groups, kneighbors: int, use_xyz=True, normalize="center"):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            # self.alpha_attr = paddle.ParamAttr(initializer=paddle.ones([1, 1, 1, channel + add_channel]))
            self.affine_alpha = paddle.create_parameter(shape=[1, 1, 1, channel + add_channel],
                                                        dtype=paddle.float32,
                                                        default_initializer=paddle.nn.initializer.Constant(1.))
            # self.beta_attr = paddle.ParamAttr(initializer=paddle.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = paddle.create_parameter(shape=[1, 1, 1, channel + add_channel],
                                                       dtype=paddle.float32,
                                                       default_initializer=paddle.nn.initializer.Constant(1.))
            self.add_parameter('alpha', self.affine_alpha)
            self.add_parameter('beta', self.affine_beta)
            # self.affine_alpha = paddle.ones(shape=[1, 1, 1, channel + add_channel], dtype=paddle.float32)
            # self.affine_beta = paddle.ones(shape=[1, 1, 1, channel + add_channel], dtype=paddle.float32)

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        fps_idx = farthest_point_sampling(xyz, self.groups)  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]
        idx = knn_point(xyz, new_xyz, self.kneighbors)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = paddle.concat([grouped_points, grouped_xyz], axis=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize == "center":
                mean = paddle.mean(grouped_points, axis=2, keepdim=True)
            else:
                mean = paddle.concat([new_points, new_xyz], axis=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(axis=-2)  # [B, npoint, 1, d+3]
            std = paddle.std((grouped_points - mean).reshape([B, -1]), axis=-1, keepdim=True).unsqueeze(axis=-1).unsqueeze(axis=-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta

        new_points = paddle.concat([grouped_points, new_points.reshape([B, S, 1, -1]).tile([1, 1, self.kneighbors, 1])],
                                   axis=-1)

        return new_xyz, new_points


class PreExtraction(paddle.nn.Layer):
    def __init__(self, channels, out_channels, blocks=1, groups=1, res_expansion=1., use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = Conv1DBlock(in_channels, out_channels)
        operation = []
        for _ in range(blocks):
            operation.append(Conv1DResBlock(out_channels, kernel_size=1, groups=groups, res_expansion=res_expansion))
        self.operation = paddle.nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.shape  # [32, 512, 32, 6]
        x = x.transpose([0, 1, 3, 2])
        x = x.reshape([-1, d, s])
        x = self.transfer(x)
        batch_size, _, _ = x.shape
        x = self.operation(x)  # [b, d, k]
        x = paddle.nn.functional.adaptive_max_pool1d(x, 1).reshape([batch_size, -1])
        x = x.reshape([b, n, -1]).transpose([0, 2, 1])

        return x


class PosExtraction(paddle.nn.Layer):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1.):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(Conv1DResBlock(channels, kernel_size=1, groups=groups, res_expansion=res_expansion))
        self.operation = paddle.nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)


class Model(paddle.nn.Layer):
    def __init__(self, points=1024, class_num=40, embed_dim=64, groups=1, res_expansion=1.0, use_xyz=True,
                 normalize="center",
                 dim_expansion=(2, 2, 2, 2), pre_blocks=(2, 2, 2, 2), pos_blocks=(2, 2, 2, 2),
                 k_neighbors=(32, 32, 32, 32), reducers=(2, 2, 2, 2)):
        super(Model, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        self.embedding = Conv1DBlock(3, embed_dim)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = paddle.nn.LayerList()
        self.pre_blocks_list = paddle.nn.LayerList()
        self.pos_blocks_list = paddle.nn.LayerList()
        last_channel = embed_dim
        anchor_points = self.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel

        self.classifier = paddle.nn.Sequential(paddle.nn.Linear(last_channel, 512),
                                               paddle.nn.BatchNorm1D(512),
                                               paddle.nn.ReLU(),
                                               paddle.nn.Dropout(0.5),
                                               paddle.nn.Linear(512, 256),
                                               paddle.nn.BatchNorm1D(256),
                                               paddle.nn.ReLU(),
                                               paddle.nn.Dropout(0.5),
                                               paddle.nn.Linear(256, self.class_num))

    def forward(self, x):
        xyz = x.transpose([0, 2, 1])
        batch_size, _, _ = x.shape
        x = self.embedding(x)  # B,D,N
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x = self.local_grouper_list[i](xyz, x.transpose([0, 2, 1]))  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = self.pos_blocks_list[i](x)  # [b,d,g]

        x = paddle.nn.functional.adaptive_max_pool1d(x, 1).squeeze(axis=-1)
        x = self.classifier(x)

        return x


def PointMLP(num_classes=40) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
                 use_xyz=False, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2])


def PointMLPElite(num_classes=40) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=32, groups=1, res_expansion=0.25,
                 use_xyz=False, normalize="anchor",
                 dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                 k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2])


if __name__ == '__main__':
    net = PointMLP()
    # inp = paddle.randn(shape=[16, 3, 1024])
    # out = net(inp)
    net_infos = paddle.summary(net, (16, 3, 1024))
    print(net_infos)
