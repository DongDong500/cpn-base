#import EarlyStopping
#from pytorchtools import EarlyStopping
import numpy as np
import torch
import os

class EarlyStopping:
    """ Stop training if validation loss or any score (Dice, Acc ...) does not improve after given patience """
    def __init__(self, patience:int = 100, delta:int = 0, verbose:bool = False,
                    path:str = 'checkpoint.pt', ceiling:bool = False, ckpt:str = 'checkpoint.pt'):
        """
        Args:
            patience (int): waits until given patience epochs after improved validation loss or scores. 
                            (Default: 100)
            delta (float): minimum change in monitored quantity to be considered as improved. 
                            (Default: 0)
            verbose (bool): if true, show the imporvement. 
                            (Default: False)
            path (str): path to checkpoint. 
                            (Default: 'checkpoint.pt')
            ceiling (bool): if true, higher score indicates improved performance.
                            (Default: False)
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.ceiling = ceiling
        self.counter = 0
        self.best_score = -np.Inf if self.ceiling else np.Inf
        self.early_stop = False
        self.ckpt = ckpt


    def __call__(self, score, model, optimizer, scheduler, cur_epoch):

        
        if self.ceiling:
            if score > self.best_score + self.delta:
                self.save_checkpoint(score, model, optimizer, scheduler, cur_epoch)
                self.counter = 0
                self.best_score = score
                return True
            else:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                return False
        else:
            if score < self.best_score - self.delta:
                self.save_checkpoint(score, model, optimizer, scheduler, cur_epoch)
                self.counter = 0
                self.best_score = score
                return True
            else:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                return False

    def save_checkpoint(self, score, model, optimizer, scheduler, cur_epoch):
        '''save model parameters if validation loss has been improved'''
        if self.verbose:
            msg = 'Score increased' if self.ceiling else 'Validation loss decreased'
            print(f'{msg} ({self.best_score:.4f} --> {score:.4f})')
        
        torch.save({
            'model_state' : model.state_dict(),
            'optimizer_state' : optimizer.state_dict(),
            'scheduler_state' : scheduler.state_dict(),
            'cur_epoch' : cur_epoch,
        }, os.path.join(self.path, self.ckpt))