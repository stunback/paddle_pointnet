import paddle


class ConvBN(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBN, self).__init__()
        self.conv = paddle.nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding='SAME')
        self.bn = paddle.nn.BatchNorm2D(out_channels)

    def forward(self, x):
        x = self.conv(x)
        y = self.bn(x)

        return y


class ConvBlock(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = paddle.nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding='SAME')
        self.bn = paddle.nn.BatchNorm2D(out_channels)
        self.relu = paddle.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        y = self.relu(x)

        return y


class ConvRedisualBlock(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bottleneck=True):
        super(ConvRedisualBlock, self).__init__()
        self.use_bottleneck = bottleneck
        if self.use_bottleneck:
            mid_channels = in_channels // 2
            self.convblock1 = ConvBlock(in_channels, mid_channels, 1, stride)
            self.convblock2 = ConvBlock(mid_channels, mid_channels, kernel_size, 1)
            self.convblock3 = ConvBN(mid_channels, out_channels, 1, 1)
        else:
            self.convblock1 = ConvBlock(in_channels, in_channels, kernel_size, stride)
            self.convblock2 = ConvBN(in_channels, out_channels, kernel_size, 1)

        if in_channels == out_channels:
            self.identity = paddle.nn.Identity()
        else:
            self.identity = ConvBN(in_channels, out_channels, 1, 1)
        self.relu = paddle.nn.ReLU()

    def forward(self, x0):
        if self.use_bottleneck:
            x1 = self.convblock1(x0)
            x1 = self.convblock2(x1)
            x1 = self.convblock3(x1)
        else:
            x1 = self.convblock1(x0)
            x1 = self.convblock2(x1)
        y = self.relu(self.identity(x0) + x1)

        return y


class Conv1DBN(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(Conv1DBN, self).__init__()
        self.conv = paddle.nn.Conv1D(in_channels, out_channels, kernel_size, stride, padding='SAME')
        self.bn = paddle.nn.BatchNorm1D(out_channels)

    def forward(self, x):
        x = self.conv(x)
        y = self.bn(x)

        return y


class Conv1DBlock(paddle.nn.Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=1, stride=1, groups=1):
        super(Conv1DBlock, self).__init__()
        self.conv = paddle.nn.Conv1D(in_channels, out_channels, kernel_size, stride, padding='SAME', groups=groups)
        self.bn = paddle.nn.BatchNorm1D(out_channels)
        self.relu = paddle.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        y = self.relu(x)

        return y


class Conv1DResBlock(paddle.nn.Layer):
    def __init__(self, in_channels: int, kernel_size=1, groups=1, res_expansion=1.0):
        super(Conv1DResBlock, self).__init__()
        mid_channels = int(in_channels * res_expansion)
        self.mlp1 = Conv1DBlock(in_channels, mid_channels, kernel_size, 1, groups)
        if groups > 1:
            self.mlp2 = paddle.nn.Sequential(Conv1DBlock(mid_channels, in_channels, kernel_size, 1, groups),
                                             Conv1DBN(in_channels, in_channels, kernel_size))
        else:
            self.mlp2 = paddle.nn.Sequential(Conv1DBN(mid_channels, in_channels, kernel_size))

        self.relu = paddle.nn.ReLU()

    def forward(self, x):
        return self.relu(self.mlp2(self.mlp1(x)) + x)

