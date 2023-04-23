import os
import math
import argparse
import time
import pandas as pd
import numpy as np
from PIL import Image
import scipy.misc

import torch
import torchvision
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch import autograd
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets
from torchvision import models
from torchvision import transforms
from torchvision.utils import make_grid

from utils import get_R2
from data_loader import CensusGoogleSatellite

# Global Variables
IMAGE_SIZE = 224
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 5
LOG_AFTER = 100
num_outputs = 16

mean = [0.4172209006283158, 0.4474928889875387, 0.4321677989697857]
std = [0.14527106771274262, 0.11944838697773438, 0.11395438658730368]


data_transforms = {
    'train': transforms.Compose([
#         transforms.CenterCrop(400),
#         transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
#         transforms.CenterCrop(400),
#         transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
#         transforms.CenterCrop(400),
#         transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

def train(args):          
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    use_cuda=False
    dtype=torch.FloatTensor
    exp_loc = os.path.join("../local/experiments",args.exp_name)
    os.system("mkdir -p "+exp_loc)
    log_file_path=os.path.join(exp_loc,"log.txt")
    log_file=open(log_file_path,"a")

    if args.gpu :
        args.gpu = torch.cuda.is_available()
    
    if (args.gpu):
        use_cuda = True
        dtype = torch.cuda.FloatTensor
        print ("Current device: %d" %torch.cuda.current_device())

    if args.model_type == 'resnet18':
        # resnet 18
        image_enc = torchvision.models.resnet18(pretrained=True)
        num_ftrs = image_enc.fc.in_features
        image_enc.fc = nn.Linear(num_ftrs,num_outputs)

    elif args.model_type == 'resnet34' :
        # resnet 34
        image_enc = torchvision.models.resnet34(pretrained=True)
        num_ftrs = image_enc.fc.in_features
        image_enc.fc = nn.Linear(num_ftrs,num_outputs)

    elif args.model_type == 'resnet50' :
        # resnet 50
        image_enc = torchvision.models.resnet50(pretrained=True)
        num_ftrs = image_enc.fc.in_features
        image_enc.fc = nn.Linear(num_ftrs,num_outputs)

    else :
        #vgg 16
        image_enc = torchvision.models.vgg16_bn(pretrained=True)
        num_ftrs = image_enc.classifier[6].in_features
        clsf = list(image_enc.classifier.children())[:-1] # Remove last layer
        clsf.extend([nn.Linear(num_ftrs,num_outputs)])
        image_enc.classifier = nn.Sequential(*clsf)

    if(args.load_wt):
        image_enc.load_state_dict(torch.load(args.model_path))
        
    if(args.gpu):
        image_enc.cuda()

    optimizer = Adam(image_enc.parameters(),args.lr) 
    loss_mse = torch.nn.MSELoss()


    train_dataset = CensusGoogleSatellite(filename=args.dataset,transform=data_transforms['train'],image_root=args.image_root)
    train_loader = DataLoader(train_dataset, batch_size=2*args.batch_size, shuffle=True)
    val_dataset = CensusGoogleSatellite(filename=args.dataset,transform=data_transforms['val'],mode='val',image_root=args.image_root)
    val_loader = DataLoader(val_dataset, batch_size=2*args.batch_size, shuffle=True)

    best_r2 = -100
    
    for e in range(args.epochs):

        # track values for...
        img_count = 0
        aggregate_loss = 0.0
        run_y = np.zeros((0,num_outputs),dtype=np.float64)
        run_y_ = np.zeros((0,num_outputs),dtype=np.float64)
        
        # train network
        image_enc.train()
        for batch_num, (x, y) in enumerate(train_loader):
            img_batch_read = len(x)
            img_count += img_batch_read

            # zero out gradients
            optimizer.zero_grad()

            # input batch to model
            x = Variable(x).type(dtype)
            y = Variable(y).type(dtype)
            
            y_hat = image_enc(x)

            current_loss=loss_mse(y,y_hat)

            aggregate_loss += current_loss
            run_y = np.concatenate((run_y,y.data.cpu().numpy()),axis=0)
            run_y_ = np.concatenate((run_y_,y_hat.data.cpu().numpy()),axis=0)

            # backprop
            current_loss.backward()
            optimizer.step()

            # print out status message
            if ((batch_num + 1) % LOG_AFTER == 0):
                run_r2 = get_R2(run_y,run_y_)
                status = "{}  Epoch {}:  [{}/{}]  Batch:[{}]  agg_loss: {:.6f}".format(
                                time.ctime(), e + 1, img_count, len(train_dataset), batch_num+1,
                                aggregate_loss/(batch_num+1.0))
                print(status)
                print("train R2: ",run_r2)
                print(status,file=log_file)
                print("train R2: ",run_r2,file=log_file)
                log_file.flush()

        

        # save model
        
        image_enc.eval()

        if use_cuda:
            image_enc.cpu()

        model_dir=exp_loc
        filename = os.path.join(model_dir,str(e)+".model")
        # filename = "models/" + str(e) + "_" + str(time.ctime()).replace(' ', '_') + ".model"
        torch.save(image_enc.state_dict(), filename)
    
        if use_cuda:
            image_enc.cuda()

        run_y = np.zeros((0,num_outputs),dtype=np.float64)
        run_y_ = np.zeros((0,num_outputs),dtype=np.float64)
        batch_count = 0
        for _, (x, y) in enumerate(val_loader):
            x = Variable(x).type(dtype)
            y = Variable(y).type(dtype)
            y_hat = image_enc(x)
            run_y = np.concatenate((run_y,y.data.cpu().numpy()),axis=0)
            run_y_ = np.concatenate((run_y_,y_hat.data.cpu().numpy()),axis=0)

        val_r2 = (get_R2(run_y,run_y_))
        val_r2_mean = np.mean(get_R2(run_y,run_y_))

        status = "{}  Epoch {}: ".format(time.ctime(), e + 1)
        print(status)
        print("val R2: ",val_r2)
        print(status,file=log_file)
        print("val R2: ",val_r2,file=log_file)
        
        print("val R2 mean :",val_r2_mean)
        print("val R2 mean :",val_r2_mean,file=log_file)
        
        log_file.flush()
        if(val_r2_mean>best_r2):
            image_enc.eval()

            if use_cuda:
                image_enc.cpu()

            model_dir=exp_loc
            filename = os.path.join(model_dir,"best.model")
            torch.save(image_enc.state_dict(), filename)
            if use_cuda:
                image_enc.cuda()
            best_r2 = val_r2_mean
            
    log_file.close()

def test(args):
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # GPU enabling
    use_cuda = False
    dtype = torch.FloatTensor
    
    if args.gpu :
        args.gpu = torch.cuda.is_available()
    
    if (args.gpu):
        use_cuda = True
        dtype = torch.cuda.FloatTensor
        # torch.cuda.set_device(args.gpu)
        print ("Current device: %d" %torch.cuda.current_device())

    if (args.gpu):
        use_cuda = True
        dtype = torch.cuda.FloatTensor
        print ("Current device: %d" %torch.cuda.current_device())

    if args.model_type == 'resnet18':
        # resnet 18
        image_enc = torchvision.models.resnet18(pretrained=True)
        num_ftrs = image_enc.fc.in_features
        image_enc.fc = nn.Linear(num_ftrs,num_outputs)

    elif args.model_type == 'resnet34' :
        # resnet 34
        image_enc = torchvision.models.resnet34(pretrained=True)
        num_ftrs = image_enc.fc.in_features
        image_enc.fc = nn.Linear(num_ftrs,num_outputs)

    elif args.model_type == 'resnet50' :
        # resnet 50
        image_enc = torchvision.models.resnet50(pretrained=True)
        num_ftrs = image_enc.fc.in_features
        image_enc.fc = nn.Linear(num_ftrs,num_outputs)

    else :
        #vgg 16
        image_enc = torchvision.models.vgg16_bn(pretrained=True)
        num_ftrs = image_enc.classifier[6].in_features
        clsf = list(image_enc.classifier.children())[:-1] # Remove last layer
        clsf.extend([nn.Linear(num_ftrs,num_outputs)])
        image_enc.classifier = nn.Sequential(*clsf)
    
    image_enc.load_state_dict(torch.load(args.model_path))
    
    
    test_dataset = data_loader.CensusGoogleSatellite(filename=args.dataset,transform=data_transforms['test'],mode='test',image_root=args.image_root)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size)

    image_enc.eval()
    loss_mse = torch.nn.MSELoss()
    counter=0
    run_y = np.zeros((0,num_outputs),dtype=np.float64)
    run_y_ = np.zeros((0,num_outputs),dtype=np.float64)
    aggregate_loss = 0.0
    batch_count = 0
    for _, (x, y) in enumerate(test_loader):
        batch_count += 1
        x = Variable(x).type(dtype)
        y = Variable(y).type(dtype)
        y_hat = image_enc(x)
        run_y = np.concatenate((run_y,y.data.cpu().numpy()),axis=0)
        run_y_ = np.concatenate((run_y_,y_hat.data.cpu().numpy()),axis=0)
        current_loss=loss_mse(y,y_hat)
        aggregate_loss += current_loss
    aggregate_loss = aggregate_loss/batch_count
    test_r2 = get_R2(run_y,run_y_)
    print("test_loss:",aggregate_loss)
    print("test_R2:",test_r2)

def main():
    parser = argparse.ArgumentParser(description='Training a CNN to predict the census')
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train", help="Training the model")
    train_parser.add_argument("--dataset", type=str, required=True, help="Path to census file")
    train_parser.add_argument("--image_root", type=str, required=True, help="Path to Images")
    train_parser.add_argument("--gpu", action='store_true', help="GPU to be used")
    train_parser.add_argument("--load_wt",action='store_true',help="load previous wts")
    train_parser.add_argument("--model_path", type=str, required=False, help="Path to network initial wts")
    train_parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    train_parser.add_argument("--model_type", type=str, required=False,default = "vgg16", help="model to be used")
    train_parser.add_argument("--batch_size", type=int, required=False,default = BATCH_SIZE, help="batch size")
    train_parser.add_argument("--epochs", type=int, required=False,default = EPOCHS, help="No. of epochs to be trained for")
    train_parser.add_argument("--seed", type=int, required=False,default = 8797, help="random seed")
    train_parser.add_argument("--lr", type=float, required=False,default = LEARNING_RATE, help="learning rate")
    
    test_parser = subparsers.add_parser("test", help="Testing the model")
    test_parser.add_argument("--model_path", type=str, required=True, help="path to a trained model")
    test_parser.add_argument("--dataset", type=str, required=True, help="Path to census file")
    test_parser.add_argument("--image_root", type=str, required=True, help="Path to Images")
    test_parser.add_argument("--gpu", action='store_true', help="GPU to be used")
    test_parser.add_argument("--model_type", type=str, required=False,default = "vgg16", help="model to be used")
    test_parser.add_argument("--batch_size", type=int, required=False,default = BATCH_SIZE, help="batch size")
    test_parser.add_argument("--seed", type=int, required=False,default = 8797, help="random seed")
    
    args = parser.parse_args()

    # command
    if (args.subcommand == "train"):
        print ("Training!")
        train(args)
    elif (args.subcommand == "test"):
        print ("testing!")
        test(args)
    else:
        print("invalid command")

if __name__ == '__main__':
    main()








