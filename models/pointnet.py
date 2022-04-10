import numpy as np
import paddle
from models.base import Conv1DBN, Conv1DBlock


class TNet(paddle.nn.Layer):
    """
    PointNet TNet
    """

    def __init__(self, in_channels=3):
        super(TNet, self).__init__()
        self.conv_block = paddle.nn.Sequential(Conv1DBlock(in_channels, 64, 1),
                                               Conv1DBlock(64, 128, 1),
                                               Conv1DBlock(128, 1024, 1),
                                               paddle.nn.AdaptiveMaxPool1D(1))
        self.flatten = paddle.nn.Flatten()
        self.mlp = paddle.nn.Sequential(paddle.nn.Linear(1024, 512),
                                        paddle.nn.ReLU(),
                                        paddle.nn.Linear(512, 256),
                                        paddle.nn.ReLU(),
                                        paddle.nn.Linear(256, in_channels ** 2))
        self.identity = paddle.eye(in_channels, dtype=paddle.float32).reshape([-1])

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.mlp(x)
        y = x + self.identity

        return y


class PointNet(paddle.nn.Layer):
    """
    PointNet structure
    """

    def __init__(self, num_classes=16):
        super(PointNet, self).__init__()
        self.input_stn = TNet(in_channels=3)
        self.mlp1 = paddle.nn.Sequential(Conv1DBlock(3, 64, 1),
                                         Conv1DBlock(64, 64, 1))
        self.feature_stn = TNet(in_channels=64)
        self.mlp2 = paddle.nn.Sequential(Conv1DBlock(64, 64, 1),
                                         Conv1DBlock(64, 128, 1),
                                         Conv1DBlock(128, 1024, 1))
        self.maxpool = paddle.nn.AdaptiveMaxPool1D(1)
        self.flatten = paddle.nn.Flatten()
        self.mlp3 = paddle.nn.Sequential(paddle.nn.Linear(1024, 512),
                                         paddle.nn.ReLU(),
                                         paddle.nn.Linear(512, 256),
                                         paddle.nn.ReLU(),
                                         paddle.nn.Dropout(0.5),
                                         paddle.nn.Linear(256, num_classes))

    def forward(self, x):
        input_trans = self.input_stn(x).reshape([-1, 3, 3])  # t_mat
        x = paddle.matmul(input_trans, x)
        x = self.mlp1(x)
        feature_trans = self.feature_stn(x).reshape([-1, 64, 64])  # t_mat
        x = paddle.matmul(feature_trans, x)
        x = self.mlp2(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        y = self.mlp3(x)

        return y, input_trans, feature_trans


if __name__ == '__main__':
    net = PointNet()
    paddle.summary(net, (32, 3, 2048))
