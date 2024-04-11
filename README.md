# Multi Early stopping 

'''
import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience, loss_names, verbose=False, path='Earlystopping.pt'):
        self.patience = patience
        self.loss_names = loss_names
        self.verbose = verbose
        self.counters = 0
        self.best_scores = {loss_name: None for loss_name in loss_names}
        self.early_stop = False
        self.loss_min = {loss_name: np.Inf for loss_name in loss_names}
        self.loss_min_past = {loss_name: np.Inf for loss_name in loss_names}
        self.path = path

    def __call__(self, model, losses):

        all_losses_improved = all(loss_value < self.loss_min[loss_name] for loss_name, loss_value in losses.items())
        
        if all_losses_improved:
            for loss_name, loss_value in losses.items():
                if self.best_scores[loss_name] is None:
                    self.best_scores[loss_name] = loss_value
                    self.loss_min[loss_name] = loss_value
                    self.save_checkpoint(model)

                elif loss_value < self.best_scores[loss_name]:
                    self.best_scores[loss_name] = loss_value
                    self.loss_min_past[loss_name] = self.loss_min[loss_name]
                    self.loss_min[loss_name] = loss_value
                    self.save_checkpoint(model)
                    self.counters = 0
        else:
            self.counters += 1
            if self.counters >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        torch.cuda.empty_cache()
'''

# Stack  
 <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/pytorch-EE4C2C?style=flat&logo=pytorch&logoColor=white"/>

# Introduction

Supports early termination based on Pareto optimality, recording when all losses are improved when multi-task learning and calculating various losses

# Laboratory

School of Industrial Management and Engineering  
College of Engineering  
Korea University  
Data Analytics and Healthcare Systems Lab.
