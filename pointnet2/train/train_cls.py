import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import os

from pointnet2.models import Pointnet2ClsMSG as Pointnet
from pointnet2.models.pointnet2_msg_cls import model_fn_decorator
from pointnet2.data import ModelNet40Cls
import pointnet2.utils.pytorch_utils as pt_utils
import pointnet2.data.data_utils as d_utils
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
        default=4096,
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
        "-decay_step", type=float, default=2e5, help="Learning rate decay step"
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
    parser.add_argument('--visdom-port', type=int, default=8097)

    return parser.parse_args()


lr_clip = 1e-5
bnm_clip = 1e-2

if __name__ == "__main__":
    args = parse_args()

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
        args.num_points, transforms=transforms, train=False
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    train_set = ModelNet40Cls(args.num_points, transforms=transforms)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    model = Pointnet(input_channels=0, num_classes=40, use_xyz=True)
    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_lbmd = lambda it: max(args.lr_decay**(int(it * args.batch_size / args.decay_step)), lr_clip / args.lr)
    bn_lbmd = lambda it: max(args.bn_momentum * args.bnm_decay**(int(it * args.batch_size / args.decay_step)), bnm_clip)

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

    viz = pt_utils.VisdomViz(port=args.visdom_port)
    viz.text(str(vars(args)))

    trainer = pt_utils.Trainer(
        model,
        model_fn,
        optimizer,
        checkpoint_name="checkpoints/pointnet2_cls",
        best_name="checkpoints/pointnet2_cls_best",
        lr_scheduler=lr_scheduler,
        bnm_scheduler=bnm_scheduler,
        viz=viz
    )

    trainer.train(
        0,
        start_epoch,
        args.epochs,
        train_loader,
        test_loader,
        best_loss=best_loss
    )

    if start_epoch == args.epochs:
        _ = trainer.eval_epoch(test_loader)
