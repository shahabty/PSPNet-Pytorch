import os
import sys
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

import torchvision.transforms as standard_transforms
import utils.transforms as extended_transforms

from utils.misc import evaluate, CrossEntropyLoss2d

import cityscapes

from pspnet import pspnet as PSPNet 
from config import pspnet_path

import torch.optim as optim
from torch.backends import cudnn
from torch.autograd import Variable
import torch.nn as nn
import torch

import torchvision.utils as vutils

from tqdm import tqdm
cudnn.benchmark = True

args = {
'val_batch_size':1,
'saved_dir':'check-points',
'gpu0':0,
'gpu1':1,
'save_as_image':False,
}

gpu0 = args['gpu0']
gpu1 = args['gpu1']

def __main__(args):
#initializing pretrained network
    pspnet = PSPNet(n_classes = cityscapes.num_classes).cuda(gpu0)
    pspnet.load_pretrained_model(model_path = pspnet_path) 
#transformation and loading dataset
    mean_std = ([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])
    val_input_transform = standard_transforms.Compose([
        extended_transforms.FlipChannels(),
        standard_transforms.ToTensor(),
        standard_transforms.Lambda(lambda x: x.mul_(255)),
        standard_transforms.Normalize(*mean_std)
    ])
     
    target_transform = standard_transforms.Compose([extended_transforms.MaskToTensor()])

    restore_transform = standard_transforms.Compose([
        extended_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage(),
    ])

    visualize = standard_transforms.ToTensor()
    val_set = cityscapes.CityScapes('val',transform = val_input_transform,target_transform = target_transform)
    val_loader = DataLoader(val_set,batch_size = args['val_batch_size'],num_workers = 8,shuffle = False)
    validate(pspnet,val_loader,cityscapes.num_classes,args,restore_transform,visualize)
    
def validate(pspnet,val_loader,num_classes,args,restore_transform,visualize):
    pspnet.eval()
    pred_all = []
    gts_all = []
    for vi,data in tqdm(enumerate(val_loader),desc = 'validation'):
        img,gts = data
        with torch.no_grad():
            img = Variable(img).cuda(gpu0)
            gts = Variable(gts).cuda(gpu0)

            output =F.softmax(pspnet(img),1)

        pred = output.data.max(1)[1].cpu().numpy()

        pred_all.append(pred)
        gts_all.append(gts.data.cpu().numpy())
 
    gts_all = np.concatenate(gts_all)
    pred_all = np.concatenate(pred_all)
    val_visual = []
    for idx,data in enumerate(pred_all):
        pred_pil = cityscapes.colorize_mask(data) 
        pred_pil.save(os.path.join('prediction', '%d_prediction.png' % idx))

    acc, acc_cls, mean_iu, fwavacc = evaluate(pred_all, gts_all,num_classes)

    print ('--------------------------------------------------------------------')
    print ('[acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (
        acc, acc_cls, mean_iu, fwavacc))

__main__(args)
