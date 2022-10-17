import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import models
import criterion
from utils import ext_transforms as et
from utils.nerve import Nerve


parser = argparse.ArgumentParser(description="nerve backbone inference")

parser.add_argument("--param_ckpt", type=str, default='/', help="parameters path")

parser.add_argument("--dst_pth", type=str, default='/')

parser.add_argument("--data_pth", type=str, default='/home/dongik/datasets',
                    help="")
parser.add_argument("--num_workers", type=int, default=8, 
                    help="number of workers (default: 8)")
parser.add_argument("--modality", type=str, default="UN", 
                    help='UN (unknown), HM (HM70A) or SN (miniSONO) (default: UN)')
parser.add_argument("--region", type=str, default="peroneal", 
                    help='peroneal, median-forearm or median-wrist (default: peroneal)')
parser.add_argument("--kfold", type=int, default=5, 
                    help="kfold (default: 5)")
parser.add_argument("--k", type=int, default=0, 
                    help="i-th fold set of kfold data (default: 0)")
parser.add_argument("--batch_size", type=int, default=16, 
                    help='test batch size (default: 16)')


if __name__ == "__main__": 
    args = parser.parse_args()
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    norm = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=norm[0], std=norm[1]),
        ])
    dst = Nerve(root_pth=args.data_pth, datatype=args.region, modality=args.modality, 
                            fold=f'v{args.kfold}/{args.k}', image_set='val', transform=transform, )
    loader = DataLoader(dst, batch_size=args.batch_size, 
                            num_workers=args.num_workers, shuffle=True, drop_last=True)

    print("Dataset - %s\n\tTest\t%d" % 
            (f'v{args.kfold}/{args.k}' + '/' + args.region, len(dst) ))

    model = models.models.__dict__['backbone_resnet50']()
    ckpt = torch.load(args.param_ckpt, map_location='cpu')
    model.load_state_dict(ckpt["model_state"])
    print(f'Best epoch: { ckpt["cur_epoch"] }')
    model.to(devices)
    model.eval()
    running_loss = 0.0
    mse = criterion.get_criterion.__dict__['mseloss']()

    with torch.no_grad():
        for i, (ims, lbls) in tqdm(enumerate(loader), total=len(loader)):

            ims = ims.to(devices)
            bbox = lbls[1].to(devices)

            outputs = model(ims)
            mse_loss = mse(outputs, bbox)
            running_loss += mse_loss.item() * ims.size(0)

        epoch_loss = running_loss / len(loader)

        print(f"Epoch loss : {epoch_loss:.6f}")
