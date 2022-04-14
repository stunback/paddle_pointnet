import os
import argparse
import datetime

import numpy as np
from tqdm import tqdm
from visualdl import LogWriter
import paddle

from models import pointnet, pointnet2, dgcnn
from data import shapenet_dataset, modelnet_dataset
from utils import model_loss


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['shapenet', 'modelnet'], help='dataset type')
    parser.add_argument('--dataset_path', type=str, required=True, help='dataset path')
    parser.add_argument('--logdir', type=str, default='logs', help='where to save training logs')
    parser.add_argument('--model', type=str, default='pointnet', help='model architecture')
    parser.add_argument('-b', '--batchsize', type=int, default=32, help='training batchsize')
    parser.add_argument('-e', '--epochs', type=int, default=30, help='training epochs')
    parser.add_argument('--optim', type=str, default='adam', help='training optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--workers', type=int, default=0, help='multiprocessing workers num')
    parser.add_argument('-r', '--resume', action='store_true', default=False,
                        help='resume training from last epoch')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint path for resume training')
    args = parser.parse_args()
    assert args.model in ['pointnet', 'pointnet2', 'dgcnn'], 'Unsupported model architecture {}'.format(args.model)
    assert args.optim in ['adam', 'sgd'], 'Unsupported optimizer {}'.format(args.optim)

    return args


def train(opts):
    # Step 0: create logger path
    if not os.path.exists(opts.logdir):
        os.mkdir(opts.logdir)
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(opts.logdir, 'trainval', 'exp' + now)
    os.makedirs(exp_dir)

    # Step 1: load train val dataset
    if opts.dataset == 'shapenet':
        loaders = shapenet_dataset.get_dataloader(opts.dataset_path, batch_size=opts.batchsize, num_workers=opts.workers)
        train_loader, val_loader = loaders
        num_classes = 16
    else:
        loaders = modelnet_dataset.get_txt_dataloader(opts.dataset_path, batch_size=opts.batchsize, num_workers=opts.workers)
        train_loader, val_loader = loaders
        num_classes = 40
    # Step 2: load model, loss, optimizer
    net = None
    if opts.model == 'pointnet':
        net = pointnet.PointNet(num_classes=num_classes)
    else:
        print('Temporarily not supported')
        exit(-1)

    # cross_entropy = paddle.nn.CrossEntropyLoss(soft_label=True)
    pointnetloss = model_loss.PointNetLoss()
    acc = paddle.metric.Accuracy()

    epochs = opts.epochs
    # lr = paddle.optimizer.lr.ReduceOnPlateau(opts.lr, factor=0.5, patience=10, threshold=0.01, verbose=True)
    lr = paddle.optimizer.lr.StepDecay(opts.lr, step_size=20, gamma=0.7, verbose=True)
    if opts.optim == 'adam':
        optim = paddle.optimizer.Adam(lr, parameters=net.parameters(), weight_decay=0.0001)
    else:
        optim = paddle.optimizer.SGD(lr, parameters=net.parameters(), weight_decay=0.0001)

    if opts.resume:
        net.set_state_dict(paddle.load(opts.checkpoint))
        optim.set_state_dict(paddle.load(opts.checkpoint.replace('.pdparams', '.pdopt')))
    '''
    try:
        if opts.optim == 'adam':
            optim = paddle.optimizer.Adam(lr, parameters=net.parameters())
        elif opts.optim == 'sgd':
            optim = paddle.optimizer.SGD(lr, parameters=net.parameters())
        else:
            raise ValueError('Not Implemented optimizer type {}'.format(opts.optim))
    except ValueError as e:
        print(e)
    '''

    # Step 3: training and validation procedure
    logger_train = LogWriter(logdir=os.path.join(exp_dir, 'train'))
    logger_val = LogWriter(logdir=os.path.join(exp_dir, 'val'))
    logger_epoch = LogWriter(logdir=os.path.join(exp_dir, 'epoch'))
    logger_acc = LogWriter(logdir=os.path.join(exp_dir, 'epoch', 'acc'))
    last_avg_acc = 0.
    for epoch in range(epochs):
        # Step 3.1: training and validation procedure
        net.train()
        train_loop = tqdm(enumerate(train_loader),
                          total=len(train_loader)
                          )
        all_batch_loss = paddle.zeros([1], dtype=paddle.float32)
        for batch_id, data in train_loop:
            x_data = data[0]
            y_data = data[1]
            label_cls = paddle.nn.functional.one_hot(y_data.squeeze(), num_classes=num_classes)
            inp = x_data
            out_cls, trans_mat3, trans_mat64 = net(inp)
            # loss = cross_entropy(out_cls, label_cls)
            loss = pointnetloss(out_cls, label_cls, trans_mat3, trans_mat64)
            loss.backward()
            optim.step()
            optim.clear_grad()

            # TODO: define a meter for loss and metrics
            all_batch_loss += loss
            avg_batch_loss = all_batch_loss.numpy()[0] / (batch_id + 1)
            batch_loss = loss.numpy()[0]

            train_loop.set_description(f'Training Epoch [{epoch}/{epochs}]')
            train_loop.set_postfix(batch_loss=batch_loss, avg_batch_loss=avg_batch_loss)
            logger_train.add_scalar(tag='train/batch_loss',
                                    value=batch_loss,
                                    step=epoch * train_loop.total + batch_id)

        lr.step()

        # Step 3.2: validate after one train epoch
        net.eval()
        val_loop = tqdm(enumerate(val_loader),
                        total=len(val_loader)
                        )
        all_batch_acc = paddle.zeros([1], dtype=paddle.float32)
        for batch_id, data in val_loop:
            x_data = data[0]
            y_data = data[1]
            with paddle.no_grad():
                label_cls = paddle.nn.functional.one_hot(y_data.squeeze(), num_classes=num_classes)
                inp = x_data
                out_cls, _, _ = net(inp)
                correct = acc.compute(out_cls, label_cls)
                acc.update(correct)

                # TODO: define a meter for loss and metrics
                all_batch_acc += paddle.mean(correct)
                avg_batch_acc = all_batch_acc.numpy()[0] / (batch_id + 1)
                batch_acc = paddle.mean(correct).numpy()[0]
                val_loop.set_description(f'Val Epoch [{epoch}/{epochs}]')
                val_loop.set_postfix(batch_acc=batch_acc, avg_batch_acc=avg_batch_acc)
                logger_val.add_scalar(tag='val/batch_acc',
                                      value=batch_acc,
                                      step=epoch * val_loop.total + batch_id)

        cls_acc = acc.accumulate()
        acc.reset()
        logger_epoch.add_scalar(tag='train/loss',
                                value=all_batch_loss.numpy()[0] / train_loop.total,
                                step=epoch)
        logger_epoch.add_scalar(tag='val/acc',
                                value=all_batch_acc.numpy()[0] / val_loop.total,
                                step=epoch)
        logger_epoch.add_scalar(tag='epoch/lr',
                                value=lr.last_lr,
                                step=epoch)
        logger_acc.add_scalar(tag='val/acc', value=cls_acc, step=epoch)
        print('val epoch: {}, cls acc:{:.3f}'.format(epoch, cls_acc))

        # Step 3.3: decide whether to save best ckpt based on validation result
        paddle.save(net.state_dict(), os.path.join(exp_dir, 'pointnet_last.pdparams'))
        paddle.save(optim.state_dict(), os.path.join(exp_dir, 'optim_last.pdopt'))
        if avg_batch_acc > last_avg_acc:
            last_avg_acc = avg_batch_acc
            paddle.save(net.state_dict(), os.path.join(exp_dir, 'pointnet_best.pdparams'))
            paddle.save(optim.state_dict(), os.path.join(exp_dir, 'optim_best.pdopt'))
            print('save best ckpt at epoch {}!'.format(epoch))


def train2(opts):
    # Step 0: create logger path
    if not os.path.exists(opts.logdir):
        os.mkdir(opts.logdir)
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(opts.logdir, 'trainval', 'exp' + now)
    os.makedirs(exp_dir)

    # Step 1: load train val dataset
    if opts.dataset == 'shapenet':
        loaders = shapenet_dataset.get_dataloader(opts.dataset_path, batch_size=opts.batchsize, num_workers=opts.workers)
        train_loader, val_loader = loaders
        num_classes = 16
    else:
        loaders = modelnet_dataset.get_txt_dataloader(opts.dataset_path, batch_size=opts.batchsize, num_workers=opts.workers)
        train_loader, val_loader = loaders
        num_classes = 40
    # Step 2: load model, loss, optimizer
    net = None
    if opts.model == 'pointnet2':
        net = pointnet2.PointNet2(num_classes=40)
    else:
        print('Temporarily not supported')
        exit(-1)

    # cross_entropy = paddle.nn.CrossEntropyLoss(soft_label=True)
    pointnet2loss = paddle.nn.CrossEntropyLoss(soft_label=True)
    acc = paddle.metric.Accuracy()

    epochs = opts.epochs
    # lr = paddle.optimizer.lr.ReduceOnPlateau(opts.lr, factor=0.5, patience=10, threshold=0.01, verbose=True)
    lr = paddle.optimizer.lr.StepDecay(opts.lr, step_size=20, gamma=0.7, verbose=True)
    if opts.optim == 'adam':
        optim = paddle.optimizer.Adam(lr, parameters=net.parameters(), weight_decay=0.0001)
    else:
        optim = paddle.optimizer.SGD(lr, parameters=net.parameters(), weight_decay=0.0001)

    if opts.resume:
        net.set_state_dict(paddle.load(opts.checkpoint))
        optim.set_state_dict(paddle.load(opts.checkpoint.replace('.pdparams', '.pdopt')))
    '''
    try:
        if opts.optim == 'adam':
            optim = paddle.optimizer.Adam(lr, parameters=net.parameters())
        elif opts.optim == 'sgd':
            optim = paddle.optimizer.SGD(lr, parameters=net.parameters())
        else:
            raise ValueError('Not Implemented optimizer type {}'.format(opts.optim))
    except ValueError as e:
        print(e)
    '''

    # Step 3: training and validation procedure
    logger_train = LogWriter(logdir=os.path.join(exp_dir, 'train'))
    logger_val = LogWriter(logdir=os.path.join(exp_dir, 'val'))
    logger_epoch = LogWriter(logdir=os.path.join(exp_dir, 'epoch'))
    logger_acc = LogWriter(logdir=os.path.join(exp_dir, 'epoch', 'acc'))
    last_avg_acc = 0.
    for epoch in range(epochs):
        # Step 3.1: training and validation procedure
        net.train()
        train_loop = tqdm(enumerate(train_loader),
                          total=len(train_loader)
                          )
        all_batch_loss = paddle.zeros([1], dtype=paddle.float32)
        for batch_id, data in train_loop:
            x_data = data[0]
            y_data = data[1]
            label_cls = paddle.nn.functional.one_hot(y_data.squeeze(), num_classes=num_classes)
            inp = x_data
            out_cls = net(inp)
            # loss = cross_entropy(out_cls, label_cls)
            loss = pointnet2loss(out_cls, label_cls)
            loss.backward()
            optim.step()
            optim.clear_grad()

            # TODO: define a meter for loss and metrics
            all_batch_loss += loss
            avg_batch_loss = all_batch_loss.numpy()[0] / (batch_id + 1)
            batch_loss = loss.numpy()[0]

            train_loop.set_description(f'Training Epoch [{epoch}/{epochs}]')
            train_loop.set_postfix(batch_loss=batch_loss, avg_batch_loss=avg_batch_loss)
            logger_train.add_scalar(tag='train/batch_loss',
                                    value=batch_loss,
                                    step=epoch * train_loop.total + batch_id)

        lr.step()

        # Step 3.2: validate after one train epoch
        net.eval()
        val_loop = tqdm(enumerate(val_loader),
                        total=len(val_loader)
                        )
        all_batch_acc = paddle.zeros([1], dtype=paddle.float32)
        for batch_id, data in val_loop:
            x_data = data[0]
            y_data = data[1]
            with paddle.no_grad():
                label_cls = paddle.nn.functional.one_hot(y_data.squeeze(), num_classes=num_classes)
                inp = x_data
                out_cls = net(inp)
                correct = acc.compute(out_cls, label_cls)
                acc.update(correct)

                # TODO: define a meter for loss and metrics
                all_batch_acc += paddle.mean(correct)
                avg_batch_acc = all_batch_acc.numpy()[0] / (batch_id + 1)
                batch_acc = paddle.mean(correct).numpy()[0]
                val_loop.set_description(f'Val Epoch [{epoch}/{epochs}]')
                val_loop.set_postfix(batch_acc=batch_acc, avg_batch_acc=avg_batch_acc)
                logger_val.add_scalar(tag='val/batch_acc',
                                      value=batch_acc,
                                      step=epoch * val_loop.total + batch_id)

        cls_acc = acc.accumulate()
        acc.reset()
        logger_epoch.add_scalar(tag='train/loss',
                                value=all_batch_loss.numpy()[0] / train_loop.total,
                                step=epoch)
        logger_epoch.add_scalar(tag='val/acc',
                                value=all_batch_acc.numpy()[0] / val_loop.total,
                                step=epoch)
        logger_epoch.add_scalar(tag='epoch/lr',
                                value=lr.last_lr,
                                step=epoch)
        logger_acc.add_scalar(tag='val/acc', value=cls_acc, step=epoch)
        print('val epoch: {}, cls acc:{:.3f}'.format(epoch, cls_acc))

        # Step 3.3: decide whether to save best ckpt based on validation result
        paddle.save(net.state_dict(), os.path.join(exp_dir, 'pointnet2_last.pdparams'))
        paddle.save(optim.state_dict(), os.path.join(exp_dir, 'optim_last.pdopt'))
        if avg_batch_acc > last_avg_acc:
            last_avg_acc = avg_batch_acc
            paddle.save(net.state_dict(), os.path.join(exp_dir, 'pointnet2_best.pdparams'))
            paddle.save(optim.state_dict(), os.path.join(exp_dir, 'optim_best.pdopt'))
            print('save best ckpt at epoch {}!'.format(epoch))


def train3(opts):
    # Step 0: create logger path
    if not os.path.exists(opts.logdir):
        os.mkdir(opts.logdir)
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(opts.logdir, 'trainval', 'exp' + now)
    os.makedirs(exp_dir)

    # Step 1: load train val dataset
    if opts.dataset == 'shapenet':
        loaders = shapenet_dataset.get_dataloader(opts.dataset_path, batch_size=opts.batchsize, num_workers=opts.workers)
        train_loader, val_loader = loaders
        num_classes = 16
    else:
        loaders = modelnet_dataset.get_txt_dataloader(opts.dataset_path, batch_size=opts.batchsize, num_workers=opts.workers)
        train_loader, val_loader = loaders
        num_classes = 40
    # Step 2: load model, loss, optimizer
    net = None
    if opts.model == 'dgcnn':
        net = dgcnn.DGCNN()
    else:
        print('Temporarily not supported')
        exit(-1)

    # cross_entropy = paddle.nn.CrossEntropyLoss(soft_label=True)
    dgcnnloss = paddle.nn.CrossEntropyLoss(soft_label=True)
    acc = paddle.metric.Accuracy()

    epochs = opts.epochs
    # lr = paddle.optimizer.lr.ReduceOnPlateau(opts.lr, factor=0.5, patience=10, threshold=0.01, verbose=True)
    # lr = paddle.optimizer.lr.StepDecay(opts.lr, step_size=20, gamma=0.7, verbose=True)
    lr = paddle.optimizer.lr.CosineAnnealingDecay(opts.lr, opts.epochs, eta_min=opts.lr, verbose=True)
    if opts.optim == 'adam':
        optim = paddle.optimizer.Adam(lr, parameters=net.parameters(), weight_decay=0.0001)
    else:
        optim = paddle.optimizer.SGD(lr, parameters=net.parameters(), weight_decay=0.0001)

    if opts.resume:
        net.set_state_dict(paddle.load(opts.checkpoint))
        optim.set_state_dict(paddle.load(opts.checkpoint.replace('.pdparams', '.pdopt')))
    '''
    try:
        if opts.optim == 'adam':
            optim = paddle.optimizer.Adam(lr, parameters=net.parameters())
        elif opts.optim == 'sgd':
            optim = paddle.optimizer.SGD(lr, parameters=net.parameters())
        else:
            raise ValueError('Not Implemented optimizer type {}'.format(opts.optim))
    except ValueError as e:
        print(e)
    '''

    # Step 3: training and validation procedure
    logger_train = LogWriter(logdir=os.path.join(exp_dir, 'train'))
    logger_val = LogWriter(logdir=os.path.join(exp_dir, 'val'))
    logger_epoch = LogWriter(logdir=os.path.join(exp_dir, 'epoch'))
    logger_acc = LogWriter(logdir=os.path.join(exp_dir, 'epoch', 'acc'))
    last_avg_acc = 0.
    for epoch in range(epochs):
        # Step 3.1: training and validation procedure
        net.train()
        train_loop = tqdm(enumerate(train_loader),
                          total=len(train_loader)
                          )
        all_batch_loss = paddle.zeros([1], dtype=paddle.float32)
        for batch_id, data in train_loop:
            x_data = data[0]
            y_data = data[1]
            label_cls = paddle.nn.functional.one_hot(y_data.squeeze(), num_classes=num_classes)
            inp = x_data
            out_cls = net(inp)
            # loss = cross_entropy(out_cls, label_cls)
            loss = dgcnnloss(out_cls, label_cls)
            loss.backward()
            optim.step()
            optim.clear_grad()

            # TODO: define a meter for loss and metrics
            all_batch_loss += loss
            avg_batch_loss = all_batch_loss.numpy()[0] / (batch_id + 1)
            batch_loss = loss.numpy()[0]

            train_loop.set_description(f'Training Epoch [{epoch}/{epochs}]')
            train_loop.set_postfix(batch_loss=batch_loss, avg_batch_loss=avg_batch_loss)
            logger_train.add_scalar(tag='train/batch_loss',
                                    value=batch_loss,
                                    step=epoch * train_loop.total + batch_id)

        lr.step()

        # Step 3.2: validate after one train epoch
        net.eval()
        val_loop = tqdm(enumerate(val_loader),
                        total=len(val_loader)
                        )
        all_batch_acc = paddle.zeros([1], dtype=paddle.float32)
        for batch_id, data in val_loop:
            x_data = data[0]
            y_data = data[1]
            with paddle.no_grad():
                label_cls = paddle.nn.functional.one_hot(y_data.squeeze(), num_classes=num_classes)
                inp = x_data
                out_cls = net(inp)
                correct = acc.compute(out_cls, label_cls)
                acc.update(correct)

                # TODO: define a meter for loss and metrics
                all_batch_acc += paddle.mean(correct)
                avg_batch_acc = all_batch_acc.numpy()[0] / (batch_id + 1)
                batch_acc = paddle.mean(correct).numpy()[0]
                val_loop.set_description(f'Val Epoch [{epoch}/{epochs}]')
                val_loop.set_postfix(batch_acc=batch_acc, avg_batch_acc=avg_batch_acc)
                logger_val.add_scalar(tag='val/batch_acc',
                                      value=batch_acc,
                                      step=epoch * val_loop.total + batch_id)

        cls_acc = acc.accumulate()
        acc.reset()
        logger_epoch.add_scalar(tag='train/loss',
                                value=all_batch_loss.numpy()[0] / train_loop.total,
                                step=epoch)
        logger_epoch.add_scalar(tag='val/acc',
                                value=all_batch_acc.numpy()[0] / val_loop.total,
                                step=epoch)
        logger_epoch.add_scalar(tag='epoch/lr',
                                value=lr.last_lr,
                                step=epoch)
        logger_acc.add_scalar(tag='val/acc', value=cls_acc, step=epoch)
        print('val epoch: {}, cls acc:{:.3f}'.format(epoch, cls_acc))

        # Step 3.3: decide whether to save best ckpt based on validation result
        paddle.save(net.state_dict(), os.path.join(exp_dir, 'dgcnn_last.pdparams'))
        paddle.save(optim.state_dict(), os.path.join(exp_dir, 'optim_last.pdopt'))
        if avg_batch_acc > last_avg_acc:
            last_avg_acc = avg_batch_acc
            paddle.save(net.state_dict(), os.path.join(exp_dir, 'dgcnn_best.pdparams'))
            paddle.save(optim.state_dict(), os.path.join(exp_dir, 'optim_best.pdopt'))
            print('save best ckpt at epoch {}!'.format(epoch))


if __name__ == '__main__':
    arguments = options()
    if arguments.model == 'pointnet':
        train(arguments)
    elif arguments.model == 'pointnet2':
        train2(arguments)
    elif arguments.model == 'dgcnn':
        train3(arguments)
