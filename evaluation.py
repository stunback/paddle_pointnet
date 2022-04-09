import os
import argparse
import datetime

import numpy as np
from tqdm import tqdm
from visualdl import LogWriter
import paddle

from models import pointnet
from data import shapenet_dataset


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='dataset root path')
    parser.add_argument('--checkpoint', type=str, required=True, help='model checkpoint path')
    parser.add_argument('--model', type=str, default='pointnet', help='model architecture')
    parser.add_argument('--logdir', type=str, default='logs', help='where to save training logs')
    parser.add_argument('--device', type=str, default='gpu', help='inference device')
    args = parser.parse_args()

    return args


def predict(opts):
    # Step 0: create logger path
    if not os.path.exists(opts.logdir):
        os.mkdir(opts.logdir)
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(opts.logdir, 'test', 'exp' + now)
    os.makedirs(exp_dir)

    # Step 1: load train val dataset
    loaders = shapenet_dataset.get_dataloader(opts.dataset, batch_size=32, num_workers=0, if_test=True)
    test_loader = loaders

    # Step 2: load model, loss, optimizer
    net = None
    if opts.model == 'pointnet':
        net = pointnet.PointNet(num_classes=16)
    else:
        print('Temporarily not supported')
        exit(-1)

    assert opts.device in ['cpu', 'gpu'], 'Unsupported device: {}!'.format(opts.device)
    paddle.device.set_device(opts.device)
    model_state_dict = paddle.load(opts.checkpoint)
    net.set_state_dict(model_state_dict)
    # paddle.jit.save(net, 'pointnet.pdmodel', input_spec=[paddle.static.InputSpec(shape=[None, 6, 180, 320], dtype='float32')])
    acc = paddle.metric.Accuracy()
    net.eval()

    # Step 3: test
    logger_acc = LogWriter(logdir=os.path.join(exp_dir, 'acc'))
    logger_test = LogWriter(logdir=os.path.join(exp_dir, 'test'))

    test_loop = tqdm(enumerate(test_loader),
                     total=len(test_loader)
                     )
    with paddle.no_grad():
        all_batch_acc = paddle.zeros([1], dtype=paddle.float32)
        for batch_id, data in test_loop:
            x_data = data[0]
            y_data = data[1]
            label_cls = paddle.nn.functional.one_hot(y_data.squeeze(), num_classes=16)
            inp = x_data
            out_cls = net(inp)
            correct = acc.compute(out_cls, label_cls)
            acc.update(correct)

            # TODO: define a meter for loss and metrics
            all_batch_acc += paddle.mean(correct)
            avg_batch_acc = all_batch_acc.numpy()[0] / (batch_id + 1)
            batch_acc = paddle.mean(correct).numpy()[0]
            test_loop.set_description(f'Test Epoch [0/0]')
            test_loop.set_postfix(batch_acc=batch_acc, avg_batch_acc=avg_batch_acc)
            logger_test.add_scalar(tag='test/batch_acc',
                                   value=batch_acc,
                                   step=batch_id)

        cls_acc = acc.accumulate()
        acc.reset()
        logger_acc.add_scalar(tag='test/acc', value=cls_acc, step=0)
        print('test epoch: {}, cls acc:{:.3f}'.format(0, cls_acc))


if __name__ == '__main__':
    arguments = options()
    predict(arguments)

