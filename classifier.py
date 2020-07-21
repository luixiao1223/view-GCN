import numpy as np
import random
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import os,shutil,json
import argparse
from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from model.view_gcn import view_GCN, SVCNN

if __name__ == '__main__':

    classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

    cnet = SVCNN('mvcnn', nclasses=40, pretraining=False, cnn_name='resnet18')
    vgcnn = view_GCN('mvcnn', cnet, nclasses=40, cnn_name='resnet18', num_views=20)

    vgcnn.eval()
    vgcnn.load("./view-gcn", "trained_view_gcn.pth")

    val_dataset = MultiviewImgDataset('data/modelnet40v2png_ori4/*/test', scale_aug=False, rot_aug=False, num_views=20,test_mode=False)

    Counter = 0
    index = 0
    for _, data in enumerate(val_dataset, 0):
        V, C, H, W = data[1].size()
        in_data = Variable(data[1]).view(-1, C, H, W).cpu()
        target = data[0]
        out_data,F1,F2=vgcnn(in_data)
        pred = torch.max(out_data, 1)[1]
        pred = pred[0].cpu().detach().numpy().tolist()
        if pred == target:
            Counter += 1
        index += 1
        print(Counter, index, Counter/len(val_dataset))
    print("total acc:", Counter/len(val_dataset))
