import numpy as np
from matplotlib import pyplot as plt
import time

from PIL import Image
import cv2

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as tfs
torch.manual_seed(42)
import albumentations as A
from albumentations.pytorch import ToTensorV2

from glob import glob

from sklearn.model_selection import train_test_split

X = []
X.extend(glob('./data/train/*.png'))
X.extend(glob('./data/train/*.jpg'))
DATA_PATH_TRAIN_LIST, DATA_PATH_TEST_LIST = train_test_split(X, 
                                                            test_size=0.1, 
                                                            random_state=42)


class TrainImageTransform(): 
    def __init__(self):
        self.im_aug = tfs.Compose([
            tfs.Resize(256),
            tfs.CenterCrop(128),
            tfs.RandomGrayscale(p=1),
            tfs.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            tfs.ToTensor(),
            tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    def __call__(self, img):
        degree = np.random.randint(360) 
        x = img.rotate(degree)
        x = self.im_aug(x)
        return x, degree
    
    
class TestImageTransform():
    
    def __init__(self):
        self.im_aug = tfs.Compose([
            tfs.Resize(256),
            tfs.CenterCrop(128),
            tfs.ToTensor(),
            tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    def __call__(self, img):
        degree = np.random.randint(360)
        x = img.rotate(degree)
        x = self.im_aug(x)
        return x, degree
    
    
class Img_Dataset(Dataset):
    
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path).convert('RGB') 
        img_transformed, angle = self.transform(img)

        return img_transformed, angle
    
    
train_val_dataset = Img_Dataset(file_list=DATA_PATH_TRAIN_LIST,
                    transform=TrainImageTransform())

train_dataset, val_dataset = random_split(train_val_dataset, [9000, 1753])

test_dataset = Img_Dataset(file_list=DATA_PATH_TEST_LIST,
                        transform=TestImageTransform())





partition = {'train': train_dataset, 'val':val_dataset, 'test':test_dataset}


cfg = {
    'VGG1M256': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 1024, 1024, 'M'],
    'VGG1M11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 'M', 512, 'M'],
    'VGG2M16': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG1M16': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 512, 'M', 1024, 1024, 'M', 1024, 1024, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class CNN(nn.Module):
    
    def __init__(self, model_code, in_channels, out_dim, act, use_bn):
        super(CNN, self).__init__()
        
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'tanh':
            self.act = nn.TanH()
        else:
            raise ValueError("Not a valid activation function code")
        
        self.layers = self._make_layers(model_code, in_channels, use_bn)
        self.fcn = nn.Sequential(nn.Conv2d(in_channels=512,
                                           out_channels=64,
                                           kernel_size=1),
                                 self.act)
        self.classifer = nn.Sequential(nn.Linear(8*8*64, 1024),
                                       self.act,
                                       nn.Dropout(p=0.3),
                                       nn.Linear(512, out_dim))
        self.gap = nn.AvgPool2d(kernel_size=4)
        
    def forward(self, x):
        x = self.layers(x)
        x = self.fcn(x)
        # x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifer(x)
        
        return x
        
    def _make_layers(self, model_code, in_channels, use_bn):
        layers = []
        for x in cfg[model_code]:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels=in_channels,
                                     out_channels=x,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)]
                if use_bn:
                    layers += [nn.BatchNorm2d(x)]
                layers += [self.act]
                in_channels = x
        return nn.Sequential(*layers)


def train(net, partition, optimizer, criterion, args):

    trainloader = DataLoader(partition['train'], 
                            batch_size=args.train_batch_size, 
                            shuffle=True, num_workers=args.num_workers)
    net.train()

    correct = 0
    total = 0
    train_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad() 

        # get the inputs
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = train_loss / len(trainloader)
    train_acc = 100 * correct / total
    return net, train_loss, train_acc

def validate(net, partition, criterion, args):

    valloader = DataLoader(partition['val'], 
                            batch_size=args.test_batch_size, 
                            shuffle=False, num_workers=args.num_workers)
    net.eval()

    correct = 0
    total = 0
    val_loss = 0 
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)

            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(valloader)
    val_acc = 100 * correct / total
    return val_loss, val_acc

def test(net, partition, args):

    testloader = DataLoader(partition['test'], 
                            batch_size=args.test_batch_size, 
                            shuffle=False, num_workers=args.num_workers)
    net.eval()
    
    correct = 0
    total = 0
    current_labels = []
    current_preds = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            current_labels.extend(labels)
            current_preds.extend(predicted)

    test_acc = 100 * correct / total
    return test_acc, current_labels, current_preds

def experiment(partition, args):
    net = CNN(model_code = args.model_code,
            in_channels = args.in_channels,
            out_dim = args.out_dim,
            act = args.act,
            use_bn = args.use_bn)
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        raise ValueError('In-valid optimizer choice')
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    min_val_loss = np.Inf
    n_epochs_stop = 10
    epochs_no_improve = 0
    early_stop = False
    iter = 0

    for epoch in range(args.epoch):  # loop over the dataset multiple times
        ts = time.time()
        net, train_loss, train_acc = train(net, partition, optimizer, criterion, args) 
        val_loss, val_acc = validate(net, partition, criterion, args) 
        te = time.time()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print('Epoch {}, Acc(train/val): {:2.2f}/{:2.2f}, Loss(train/val) {:2.2f}/{:2.2f}. Took {:2.2f} sec'.format(epoch, train_acc, val_acc, train_loss, val_loss, te-ts))
        
        if val_loss < min_val_loss:
        # Save the model
            save_model_path = f'./Original_2.pt' # {str(args.l2).split(".")[1]}-
            torch.save(net.state_dict(), save_model_path) 
            epochs_no_improve = 0
            min_val_loss = val_loss

        else:
            epochs_no_improve += 1
        iter += 1
        if epoch > 9 and epochs_no_improve == n_epochs_stop:
            print('Early stopping!' )
            early_stop = True
            break
        else:
            continue

    test_acc, current_labels, current_preds = test(net, partition, args)
    
    result = {}
    result['train_losses'] = train_losses
    result['val_losses'] = val_losses
    result['train_accs'] = train_accs
    result['val_accs'] = val_accs
    result['train_acc'] = train_acc
    result['val_acc'] = val_acc
    result['test_acc'] = test_acc

    result['test_labels'] = current_labels
    result['test_preds'] = current_preds

    return vars(args), result



import argparse
from copy import deepcopy
import multiprocessing as mp 
from multiprocessing import freeze_support

# ====== Random seed Initialization ====== #
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
args = parser.parse_args("")
args.exp_name = "l2_bn"

# ====== Model ====== #
args.model_code = 'VGG11'
args.in_channels = 3
args.out_dim = 360
args.act = 'relu'

# ====== Regularization ======= #
args.l2 = 0.00001
args.use_bn = True

# ====== Optimizer & Training ====== #
args.optim = 'Adam' #'RMSprop' #SGD, RMSprop, ADAM...
args.lr = 0.0001
args.epoch = 350

args.num_workers = 4
args.train_batch_size = 32
args.test_batch_size = 32

# ====== Experiment Variable ====== #
name_var1 = 'model_code'
name_var2 = 'optim'
list_var1 = ['VGG13']
list_var2 = ['Adam']

if __name__ == '__main__':
    mp.freeze_support()
    for var1 in list_var1:
        for var2 in list_var2:
            setattr(args, name_var1, var1)
            setattr(args, name_var2, var2)

            setting, result = experiment(partition, deepcopy(args))