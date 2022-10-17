import os

import torch.utils.data as data
from torch.utils.data import DataLoader

from . import ext_transforms as et
from .nerve import Nerve

def get_loader(args, ):

    norm = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=norm[0], std=norm[1]),
        ])
    val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=norm[0], std=norm[1]),
        ])
    test_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=norm[0], std=norm[1]),
        ])

    if args.run_test:
        test_dst = Nerve(root_pth=args.data_pth, datatype=args.region, modality=args.modality, 
                            fold=f'v{args.kfold}/{args.k}', image_set='val', transform=test_transform, )
        test_loader = DataLoader(test_dst, batch_size=args.test_batch_size, 
                                    num_workers=args.num_workers, shuffle=True, drop_last=True)
        loader = [test_loader]
        print("Dataset - %s\n\tTest\t%d" % 
                (f'v{args.kfold}/{args.k}' + '/' + args.region, len(test_dst) ))
    else:
        train_dst = Nerve(root_pth=args.data_pth, datatype=args.region, modality=args.modality, 
                            fold=f'v{args.kfold}/{args.k}', image_set='train', transform=train_transform, )
        val_dst = Nerve(root_pth=args.data_pth, datatype=args.region, modality=args.modality, 
                            fold=f'v{args.kfold}/{args.k}', image_set='val', transform=val_transform, )
        train_loader = DataLoader(train_dst, batch_size=args.train_batch_size, 
                                    num_workers=args.num_workers, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dst, batch_size=args.val_batch_size, 
                                    num_workers=args.num_workers, shuffle=True, drop_last=True)
        loader = [train_loader, val_loader]
        print("Dataset - %s\n\tTrain\t%d\n\tVal\t%d" % 
                (f'v{args.kfold}/{args.k}' + '/' + args.region, len(train_dst), len(val_dst) ))

    return loader