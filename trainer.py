import os
import random
import numbers
import numpy as np

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from datetime import datetime

import criterion
import models
import utils


def print_result(phase, score, epoch, total_itrs, loss):
    print("[{}] Epoch: {}/{} Loss: {:.5f}".format(phase, epoch, total_itrs, loss))
    print("\tF1 [0]: {:.5f} [1]: {:.5f}".format(score['Class F1'][0], score['Class F1'][1]))
    print("\tIoU[0]: {:.5f} [1]: {:.5f}".format(score['Class IoU'][0], score['Class IoU'][1]))
    print("\tOverall Acc: {:.3f}, Mean Acc: {:.3f}".format(score['Overall Acc'], score['Mean Acc']))

def add_writer_scalar(writer, phase, score, loss, epoch):
    writer.add_scalar(f'IoU BG/{phase}', score['Class IoU'][0], epoch)
    writer.add_scalar(f'IoU Nerve/{phase}', score['Class IoU'][1], epoch)
    writer.add_scalar(f'Dice BG/{phase}', score['Class F1'][0], epoch)
    writer.add_scalar(f'Dice Nerve/{phase}', score['Class F1'][1], epoch)
    writer.add_scalar(f'epoch loss/{phase}', loss, epoch)

def set_optim(args, model, backbone):

    optimizer = [None, None]
    scheduler = [None, None]
    ### Optimizer (Segmentation)
    if args.model.startswith("deeplab"):
        if args.optim == "SGD":
            optimizer[0] = torch.optim.SGD(params=[
            {'params': model.encoder.parameters(), 'lr': 0.1 * args.lr},
            {'params': model.decoder.parameters(), 'lr': args.lr},
            ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay )
        elif args.optim == "RMSprop":
            optimizer[0] = torch.optim.RMSprop(params=[
            {'params': model.backbone.parameters(), 'lr': 0.1 * args.lr},
            {'params': model.classifier.parameters(), 'lr': args.lr},
            ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay )
        elif args.optim == "Adam":
            optimizer[0] = torch.optim.Adam(params=[
            {'params': model.backbone.parameters(), 'lr': 0.1 * args.lr},
            {'params': model.classifier.parameters(), 'lr': args.lr},
            ], lr=args.lr, betas=(0.9, 0.999), eps=1e-8 )
        else:
            raise NotImplementedError
    else:
        if args.optim == "SGD":
            optimizer[0] = torch.optim.SGD(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum )
        elif args.optim == "RMSprop":
            optimizer[0] = torch.optim.RMSprop(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum )
        elif args.optim == "Adam":
            optimizer[0] = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum )
        else:
            raise NotImplementedError
    ### Optimizer (Backbone)
    optimizer[1] = torch.optim.SGD(
                    backbone.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum )

    ### Scheduler (Segmentation)
    if args.lr_policy == 'lambdaLR':
        scheduler[0] = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer[0], )
    elif args.lr_policy == 'multiplicativeLR':
        scheduler[0] = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer[0], )
    elif args.lr_policy == 'stepLR':
        scheduler[0] = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer[0], step_size=args.step_size )
    elif args.lr_policy == 'multiStepLR':
        scheduler[0] = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer[0], )
    elif args.lr_policy == 'exponentialLR':
        scheduler[0] = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer[0], )
    elif args.lr_policy == 'cosineAnnealingLR':
        scheduler[0] = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer[0], )
    elif args.lr_policy == 'cyclicLR':
        scheduler[0] = torch.optim.lr_scheduler.CyclicLR(
            optimizer=optimizer[0], )
    else:
        raise NotImplementedError
    ### Scheduler (backbone)
    scheduler[1] = torch.optim.lr_scheduler.StepLR(
                    optimizer=optimizer[1], step_size=args.step_size )

    return optimizer, scheduler

def get_pnt(pnt, crop_size):
    # Height
    if pnt[0] >= crop_size[0]/2 and (640 - pnt[0]) >= crop_size[0]/2:
        lt = (int(pnt[0] - crop_size[0]/2), 0)
        rb = (int(pnt[0] + crop_size[0]/2), 0)
    elif pnt[0] < crop_size[0]/2 and (640 - pnt[0]) >= crop_size[0]/2:
        lt = (0, 0)
        rb = (crop_size[0], 0)
    elif pnt[0] >= crop_size[0]/2 and (640 - pnt[0]) < crop_size[0]/2:
        lt = (640 - crop_size[0], 0)
        rb = (640, 0)
    lt = list(lt)
    rb = list(rb)
    # Width
    if pnt[1] >= crop_size[1]/2 and (640 - pnt[1]) >= crop_size[1]/2:
        lt[1] = int(pnt[1] - crop_size[1]/2)
        rb[1] = int(pnt[1] + crop_size[1]/2)
    elif pnt[1] < crop_size[1]/2 and (640 - pnt[1]) >= crop_size[1]/2:
        lt[1] = 0
        rb[1] = crop_size[1]
    elif pnt[1] >= crop_size[1]/2 and (640 - pnt[1]) < crop_size[1]/2:
        lt[1] = 640 - crop_size[1]
        rb[1] = 640

    return lt, rb

def crop(ims, mas, anchor, devices, crop_size=256):
    if isinstance(crop_size, numbers.Number):
        crop_size = (int(crop_size), int(crop_size))
    else:
        crop_size = crop_size
    
    cims = torch.zeros((ims.shape[0], 3, crop_size[0], crop_size[1]), 
                        device=devices, dtype=ims.dtype, )
    cmas = torch.zeros((mas.shape[0], crop_size[0], crop_size[1]), 
                        device=devices, dtype=torch.long,)
    anchor = ((anchor * 640) + 320).type(torch.int32)
    
    for i in range(ims.size()[0]):
        lt, rb = get_pnt(anchor[i], crop_size)
        cims[i, ...] = ims[i, ...][... , lt[0]:rb[0], lt[1]:rb[1]]
        cmas[i, ...] = mas[i, ...][lt[0]:rb[0], lt[1]:rb[1]]
        
    return cims, cmas

def recover(mas, cmas, anchor, devices, crop_size=256):
    if isinstance(crop_size, numbers.Number):
        crop_size = (int(crop_size), int(crop_size))
    else:
        crop_size = crop_size

    result = torch.zeros(mas.shape, device=devices, dtype=torch.long, )
    anchor = ((anchor * 640) + 320).type(torch.int32)

    for i in range(mas.size()[0]):
        lt, rb = get_pnt(anchor[i], crop_size)
        result[i, ...][lt[0]:rb[0], lt[1]:rb[1]] = cmas[i, ...]
    
    return result

def train_epoch(devices, model, backbone, loader, optimizer, scheduler, metrics, args):

    model.train()
    backbone.train()
    metrics.reset()
    running_loss = [0.0, 0.0]

    loss_func = criterion.get_criterion.__dict__[args.loss_type]()
    mse = criterion.get_criterion.__dict__['mseloss']()

    for i, (ims, lbls) in tqdm(enumerate(loader), total=len(loader)):
        optimizer[0].zero_grad()
        optimizer[1].zero_grad()

        ims = ims.to(devices)
        mas = lbls[0].to(devices)
        bbox = lbls[1].to(devices)

        anchor = backbone(ims)
        mse_loss = mse(anchor, bbox)
        mse_loss.backward()

        cims, cmas = crop(ims, mas, bbox, devices, crop_size=256)

        outputs = model(cims)
        probs = nn.Softmax(dim=1)(outputs)
        preds = recover(mas, torch.max(probs, 1)[1], bbox, devices, crop_size=256)
        true = mas.detach().cpu().numpy()

        loss = loss_func(outputs, cmas)
        loss.backward()
        optimizer[0].step()
        optimizer[1].step()
        metrics.update(true, preds.detach().cpu().numpy())

        running_loss[0] += loss.item() * ims.size(0)
        running_loss[1] += mse_loss.item() * ims.size(0)
    scheduler[0].step()
    scheduler[1].step()
    epoch_loss = [running_loss[0] / len(loader.dataset), running_loss[1] / len(loader.dataset)]
    score = metrics.get_results()

    return epoch_loss, score

def val_epoch(devices, model, backbone, loader, metrics, args):

    model.eval()
    backbone.eval()
    metrics.reset()
    running_loss = [0.0, 0.0]

    loss_func = criterion.get_criterion.__dict__[args.loss_type]()
    mse = criterion.get_criterion.__dict__['mseloss']()

    with torch.no_grad():
        for i, (ims, lbls) in tqdm(enumerate(loader), total=len(loader)):

            ims = ims.to(devices)
            mas = lbls[0].to(devices)
            bbox = lbls[1].to(devices)

            anchor = backbone(ims)
            mse_loss = mse(anchor, bbox)

            cims, cmas = crop(ims, mas, bbox, devices, crop_size=256)

            outputs = model(cims)
            probs = nn.Softmax(dim=1)(outputs)
            preds = recover(mas, torch.max(probs, 1)[1], bbox, devices, crop_size=256)
            true = mas.detach().cpu().numpy()
            
            loss = loss_func(outputs, cmas)
            metrics.update(true, preds.detach().cpu().numpy())

            running_loss[0] += loss.item() * ims.size(0)
            running_loss[1] += mse_loss.item() * ims.size(0)
        epoch_loss = [running_loss[0] / len(loader.dataset), running_loss[1] / len(loader.dataset)]
        score = metrics.get_results()

    return epoch_loss, score

def run_training(args, RUN_ID, DATA_FOLD) -> dict:
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    start_time = datetime.now()
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=os.path.join(args.TB_pth, RUN_ID, DATA_FOLD))

    ### Load datasets
    loader = utils.get_loader(args, )

    ### Load model
    model = models.models.__dict__[args.model]()
    backbone = models.models.__dict__['backbone_resnet50']()

    ### Set up optimizer and scheduler
    optimizer, scheduler = set_optim(args, model, backbone)

    ### Resume models, schedulers and optimizer
    if args.resume:
        raise NotImplementedError
    else:
        print("[!] Train from scratch...")
        resume_epoch = 0
    
    if torch.cuda.device_count() > 1:
        print('cuda multiple GPUs')
        model = nn.DataParallel(model)
        backbone = nn.DataParallel(backbone)
    model.to(devices)
    backbone.to(devices)

    ### Set up metrics
    metrics = utils.StreamSegMetrics(n_classes=2)
    early_stop = utils.EarlyStopping(patience=args.patience, delta=args.delta, verbose=True, 
                    path=os.path.join(args.BP_pth, RUN_ID, DATA_FOLD), ceiling=True, )
    backbone_stop = utils.EarlyStopping(patience=args.patience, delta=args.delta, verbose=True, 
                        path=os.path.join(args.BP_pth, RUN_ID, DATA_FOLD), ceiling=False, ckpt='backbone.pt')
    ### Train
    for epoch in range(resume_epoch, args.total_itrs):
        epoch_loss, score = train_epoch(devices, model, backbone, loader[0], optimizer, scheduler, metrics, args)
        print_result('train', score, epoch, args.total_itrs, epoch_loss[0])
        add_writer_scalar(writer, 'train', score, epoch_loss[0], epoch)
        print(f"MSE Loss: {epoch_loss[1]:.5f}")
        writer.add_scalar('MSE epoch loss/train', epoch_loss[1], epoch)

        epoch_loss, score = val_epoch(devices, model, backbone, loader[1], metrics, args)
        print_result('val', score, epoch, args.total_itrs, epoch_loss[0])
        add_writer_scalar(writer, 'val', score, epoch_loss[0], epoch)
        print(f"MSE Loss: {epoch_loss[1]:.5f}")
        writer.add_scalar('MSE epoch loss/val', epoch_loss[1], epoch)

        if early_stop(score['Class F1'][1], model, optimizer[0], scheduler[0], epoch):
            best_score = score
            best_loss = epoch_loss

        if backbone_stop(epoch_loss[1], backbone, optimizer[1], scheduler[1], epoch):
            pass

        if early_stop.early_stop:
            print("Early Stop !!!")
            break

        if args.run_demo and epoch >= 1:
            print("Run Demo !!!")
            break
    
    results = {
        "F1 score" : {
            "background" : best_score['Class F1'][0],
            "RoI" : best_score['Class F1'][1]
        },
        "Bbox regression" : {
            "MSE" : best_loss[1],
        },
        "time elapsed" : str(datetime.now() - start_time)
    }

    return results