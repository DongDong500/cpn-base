import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

class Nerve(data.Dataset):
    """
    Args:
        root_pth (string): Root directory path of the ultrasound peripheral nerve dataset.
        datatype (string): Dataset type. ``peroneal, median-forearm or median-wrist``
        modality (string): Ultrasound modality. ``UN (unknown), HM (HM70A) or SN (miniSONO)``
        fold     (string): Data fold, kfold with i-th set. E.g. ``v5/3``
        image_set (string): Select the image_set to use. ``train or val``
        transform (callable, optional): A function/transform that  takes in an PIL image
                                        and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    def get_anchor(self, target):
        size = target.size #(width, height)
        mask = np.array(target, dtype=np.uint8)
        h, w = np.where(mask > 0)
        tl = (h.min(), w.min())
        rb = (h.max(), w.max())
        pnt = np.array([ (tl[0] + rb[0] - 640) / 1280 , (tl[1] + rb[1] - 640) / 1280 ], dtype=np.float32)

        return pnt

    def _read(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        if not os.path.exists(self.images[index]):
            raise FileNotFoundError
        if not os.path.exists(self.masks[index]):
            raise FileNotFoundError
        
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index]).convert('L')                  

        assert( img.size == target.size == (640, 640) )

        target = (target, self.get_anchor(target))

        return img, target

    def __init__(self, root_pth, datatype='peroneal', modality='UN', fold='v5/3',
                    image_set='train', transform=None, ):

        self.root_pth = root_pth
        self.datatype = datatype
        self.modality = modality
        self.fold = fold
        self.image_set = image_set
        self.transform = transform

        image_dir = os.path.join(self.root_pth, self.datatype, self.modality, 'Images')
        mask_dir = os.path.join(self.root_pth, self.datatype, self.modality, 'Masks')
        split_f = os.path.join(self.root_pth, self.datatype, self.modality, 'splits',
                                self.fold, self.image_set.rstrip('\n') + '.txt')

        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            raise Exception('Dataset not found or corrupted.')
        if not os.path.exists(split_f):
            raise Exception('Wrong image_set entered!' 
                            'Please use image_set="train" or image_set="val"', split_f)

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        fileF = os.listdir(image_dir)[-1].split('.')[-1]
        self.images = [os.path.join(image_dir, x + f".{fileF}") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + f".{fileF}") for x in file_names]
        
        assert (len(self.images) == len(self.masks))

        self.image = []
        self.mask = []
        for index in range(len(self.images)):
            img, tar = self._read(index)
            self.image.append(img)
            self.mask.append(tar)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = self.image[index]
        target = self.mask[index][0]
        anchor = self.mask[index][1]
        
        if self.transform is not None:
            img, target = self.transform(img, target)
        
        return img, (target, torch.from_numpy(anchor))

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    import sys
    print(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import ext_transforms as et
    
    transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
    image_set_type = ['train', 'val']
    for ist in image_set_type:
        dst = Nerve(root_pth='/home/dongik/datasets', datatype='peroneal', 
                        modality='UN', fold='v5/3', image_set=ist, transform=transform, )
        loader = DataLoader(dst, batch_size=16, shuffle=True, num_workers=2, drop_last=True)
        print(f'len [{ist}]: {len(dst)}')

        for i, (ims, lbls) in tqdm(enumerate(loader)):
            if i == 0:
                print(lbls[1].size())
                print(lbls[0].size())
            pass
        print('Clear !!!')
    