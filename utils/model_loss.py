import paddle


class PointNetLoss(paddle.nn.Layer):
    """
    PointNet Loss
    mse + transformation mat regularization
    """
    def __init__(self, scale=0.001):
        super(PointNetLoss, self).__init__()
        self.scale = scale
        self.ce = paddle.nn.CrossEntropyLoss(soft_label=True)
        self.m3r = MatRegularizationLoss(3)
        self.m64r = MatRegularizationLoss(64)

    def forward(self, pred, label, x_mat3, x_mat64):
        mse_loss = self.ce(pred, label)
        m3r_loss = self.m3r(x_mat3)
        m64r_loss = self.m64r(x_mat64)
        loss = mse_loss + self.scale * (m3r_loss + m64r_loss)

        return loss


class MatRegularizationLoss(paddle.nn.Layer):
    """
    Mat Regularization Loss
    """
    def __init__(self, in_channels=3):
        self.in_channels = in_channels
        super(MatRegularizationLoss, self).__init__()

    def forward(self, x):
        x_mat = x.reshape([-1, self.in_channels, self.in_channels])
        target = paddle.eye(self.in_channels)
        y = paddle.norm(target - paddle.matmul(x_mat, x_mat.T), p='fro', axis=[1, 2])
        loss = paddle.mean(y)

        return loss
