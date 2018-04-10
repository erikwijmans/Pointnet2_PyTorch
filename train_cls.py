import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import os
import tensorboard_logger as tb_log

from models import Pointnet2ClsMSG as Pointnet
from models.pointnet2_msg_cls import model_fn_decorator
from data import ModelNet40Cls
import utils.pytorch_utils as pt_utils
import data.data_utils as d_utils
import argparse

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for cls training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "-num_points",
        type=int,
        default=1024,
        help="Number of points to train with"
    )
    parser.add_argument(
        "-weight_decay",
        type=float,
        default=1e-5,
        help="L2 regularization coeff"
    )
    parser.add_argument(
        "-lr", type=float, default=1e-2, help="Initial learning rate"
    )
    parser.add_argument(
        "-lr_decay", type=float, default=0.7, help="Learning rate decay gamma"
    )
    parser.add_argument(
        "-decay_step", type=int, default=20, help="Learning rate decay step"
    )
    parser.add_argument(
        "-bn_momentum",
        type=float,
        default=0.5,
        help="Initial batch norm momentum"
    )
    parser.add_argument(
        "-bnm_decay",
        type=float,
        default=0.5,
        help="Batch norm momentum decay gamma"
    )
    parser.add_argument(
        "-checkpoint", type=str, default=None, help="Checkpoint to start from"
    )
    parser.add_argument(
        "-epochs", type=int, default=200, help="Number of epochs to train for"
    )
    parser.add_argument(
        "-run_name",
        type=str,
        default="cls_run_1",
        help="Name for run in tensorboard_logger"
    )

    return parser.parse_args()


lr_clip = 1e-5
bnm_clip = 1e-2

if __name__ == "__main__":
    args = parse_args()

    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    transforms = transforms.Compose([
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudScale(),
        d_utils.PointcloudRotate(),
        d_utils.PointcloudRotatePerturbation(),
        d_utils.PointcloudTranslate(),
        d_utils.PointcloudJitter(),
        d_utils.PointcloudRandomInputDropout()
    ])

    test_set = ModelNet40Cls(
        args.num_points, BASE_DIR, transforms=transforms, train=False
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    train_set = ModelNet40Cls(args.num_points, BASE_DIR, transforms=transforms)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    tb_log.configure('runs/{}'.format(args.run_name))

    model = Pointnet(input_channels=3, num_classes=40)
    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_lbmd = lambda e: max(args.lr_decay**(e // args.decay_step), lr_clip / args.lr)
    bn_lbmd = lambda e: max(args.bn_momentum * args.bnm_decay**(e // args.decay_step), bnm_clip)

    if args.checkpoint is not None:
        start_epoch, best_loss = pt_utils.load_checkpoint(
            model, optimizer, filename=args.checkpoint.split(".")[0]
        )

        lr_scheduler = lr_sched.LambdaLR(
            optimizer, lr_lambda=lr_lbmd, last_epoch=start_epoch
        )
        bnm_scheduler = pt_utils.BNMomentumScheduler(
            model, bn_lambda=bn_lbmd, last_epoch=start_epoch
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = pt_utils.BNMomentumScheduler(model, bn_lambda=bn_lbmd)

        best_loss = 1e10
        start_epoch = 1

    model_fn = model_fn_decorator(nn.CrossEntropyLoss())

    trainer = pt_utils.Trainer(
        model,
        model_fn,
        optimizer,
        checkpoint_name="checkpoints/single_layer",
        best_name="checkpoints/single_layer_best",
        lr_scheduler=lr_scheduler,
        bnm_scheduler=bnm_scheduler
    )

    trainer.train(
        start_epoch,
        args.epochs,
        train_loader,
        test_loader,
        best_loss=best_loss
    )

    if start_epoch == args.epochs:
        _ = trainer.eval_epoch(start_epoch, test_loader)
