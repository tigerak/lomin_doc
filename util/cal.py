
from PIL import Image
from numpy import partition

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tfs




class CNN(nn.Module):
    
    def __init__(self, model_code, in_channels, out_dim, use_bn):
        super(CNN, self).__init__()
        
        self.act = nn.ReLU()
        
        self.layers = self._make_layers(model_code, 
                                        in_channels, 
                                        use_bn)
        self.classifer = nn.Sequential(nn.Linear(8192, 1024),
                                       self.act,
                                       nn.Linear(1024, out_dim))
        
    def forward(self, x):
        x = self.layers(x)
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


#################################################################
cfg = {
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = CNN('VGG13', 3, 360, True)
model.to(device)

model.load_state_dict(torch.load(r'D:/lomin/weight/128-0001-13-177-7690.pt'))

#################################################################


### 이미지 전처리 ###
class ProImageTransform:
    def __init__(self):
        self.im_aug = tfs.Compose([
            tfs.Resize(256),
            tfs.CenterCrop(128),
            tfs.ToTensor(),
            tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    def __call__(self, img):

        x = self.im_aug(img)
        return x
    

### DataSet 구성 ###
class Pro_Img_Dataset(Dataset):
    
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path).convert('RGB') # 픽셀당 추가 알파 채널이 있다고 생각 하므로 채널이 3개가 아닌 4개입니다.
        img_transformed = self.transform(img)

        return img_transformed, img_path

### 분석기 ###
class calculator:
    def __init__(self):
        pass
    
    def forward(self, file_name):
        file_name = [file_name]
        pro_dataset = Pro_Img_Dataset(file_list=file_name,
                                      transform=ProImageTransform())
        pro_dataloader = DataLoader(pro_dataset,
                                    batch_size=64)
        partition = {}
        partition['pro'] = pro_dataloader
        
        return self.pro(partition)
        
    def pro(self, partition):
        
        model.eval()
        
        current_preds = []
        img_path_list = []
        with torch.no_grad():
            for data in partition['pro']:
                images, img_path = data
                images = images.cuda()

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                current_preds.extend(predicted)
                img_path_list.extend(img_path)

        return current_preds, img_path_list