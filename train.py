import albumentations as A
import torch.utils.data as data
import scipy.io as io
import numpy as np
from torch.utils.data import DataLoader
import torch
import argparse
import torch.nn as nn
from UNet import UNet
import torchvision.transforms as transforms
import pdb # pdb.set_trace() // 커맨드: n, d, q
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2,3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CombinedDataset(data.Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def __getitem__(self, index):
        image = self.data1[index]
        label = self.data2[index]
        return image, label

    def __len__(self):
        return len(self.data1)

class Train():
    def __init__(self):
        self.EPOCHS, self.BATCH, self.STD, self.LR, self.DATA_DIR, self.CH_DIR = self.__get_args__()
        self.train_dataloader, self.val_dataloader = self.load_data()
        self.use_cuda = torch.cuda.is_available()
        
    def __get_args__(self):
        parser = argparse.ArgumentParser(description='Parameters')
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--batch', type=int, default=64)
        parser.add_argument('--std', type=float, default=0.1, help='Standard Deviation on Gaussian Noise')
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--data_dir', type=str, default='./data')
        parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
        args = parser.parse_args()
        return args.epochs, args.batch, args.std, args.learning_rate, args.data_dir, args.checkpoint_dir


    def data_augmentations(self, data1, data2):
        augmentation = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            ],
            additional_targets={'target_image':'image'}) # input과 label을 한쌍으로 augmentation
        #[19,96,160]
        data1 = data1.permute(1,2,0) #[96,160,19]
        data2 = data2.permute(1,2,0)
        data1 = data1.numpy()
        data2 = data2.numpy()
        data = augmentation(image = data1, target_image = data2)
        data1 = data['image']
        data2 = data['target_image']
        data1 = torch.Tensor(data1)
        data2 = torch.Tensor(data2)
        data1 = data1.permute(2,0,1)
        data2 = data2.permute(2,0,1)
        return data1, data2
    
    def load_data(self):
        ################################################################
        # Train dataset
        ################################################################
        print("################################")
        print("Train_X")
        X = torch.load('/SSD5_8TB/LJH/DWI_data/train/train_X_sorted.pt')
        print(X.shape) #[2087,96,160,19]  // 2087:데이터 수(slice)  96:x  160:y  19:channel 
        X = X.permute(0,3,1,2)
        print(torch.max(X))
        print(torch.min(X))
        X = (X-torch.min(X))/(torch.max(X)-torch.min(X))*2
        print(torch.max(X))
        print(torch.min(X))
        print(X.shape)
        
        print("Train_Y")
        Y = torch.load('/SSD5_8TB/LJH/DWI_data/train_LLR/train_Y_sorted.pt')
        print(Y.shape)
        Y = Y.permute(0,3,1,2)
        print(torch.max(Y))
        print(torch.min(Y))
        Y = (Y-torch.min(Y))/(torch.max(Y)-torch.min(Y))*2
        print(torch.max(Y))
        print(torch.min(Y))

        print("Doing augmentation...")
        # [2087,19,96,160]을 Augmentation 할 때. 
        # 1)데이터 1개씩 augmentation. 2)channel end 방식으로 3)numpy로
        for i in range(X.size()[0]):
            X[i,:,:,:], Y[i,:,:,:] = self.data_augmentations(X[i,:,:,:], Y[i,:,:,:])
        
        train_dataset = CombinedDataset(X.float(), Y.float())
        
        
        ################################################################
        # Valid dataset
        ################################################################
        print("################################")
        print("Valid_X")
        f = io.loadmat('/SSD5_8TB/LJH/DWI_data/val/img_recon_cor_EX5557_p105.mat')
        img_all = None
        
        if img_all is None:
            img_all = f.get('img_all')

        print(img_all.shape) # [96,160,9,21,19]
        X = np.complex64(img_all[:,:,0,:,:])
        X = torch.tensor(X)
        X = X.permute(2,3,0,1) # [21, 19, 96, 160]  // 21: 데이터 수
        X = torch.sqrt(torch.pow(X.real,2) + torch.pow(X.imag,2)) # magnitude 구하기
        print(X.shape)
        
        print("Valid_Y")
        f = io.loadmat('/SSD5_8TB/LJH/DWI_data/val_LLR/LLR_cor_EX5557_p105.mat')
        img_all = None
        
        if img_all is None:
            img_all = f.get('denoised_img')

        print(img_all.shape)
        Y = np.complex64(img_all)
        Y = torch.tensor(Y)
        Y = Y.permute(2,3,0,1)
        Y = torch.sqrt(torch.pow(Y.real,2) + torch.pow(Y.imag,2))
        print(Y.shape)
        
        X = (X-torch.min(X))/(torch.max(X)-torch.min(X))*2
        Y = (Y-torch.min(Y))/(torch.max(Y)-torch.min(Y))*2
        
        val_dataset = CombinedDataset(X.float(), Y.float())
        
        
        ################################################################
        # Loader
        ################################################################
        train_dataloader = DataLoader(train_dataset, batch_size=self.BATCH, num_workers=4, shuffle=True) # shuffle: 데이터 섞어서 과적합 방지
        val_dataloader = DataLoader(val_dataset, batch_size=self.BATCH, num_workers=4, shuffle=True)
        return train_dataloader, val_dataloader
    
    def get_model(self):
        model = UNet(in_channels=19, out_channels=19).float()
        if self.use_cuda:
            model = nn.DataParallel(model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR)
        criterion = MSELoss()
        return model, optimizer, criterion
        
    def train(self):
        model, optimizer, criterion = self.get_model()
        
        train_loss =[]
        val_loss= []
        epochs=[]    
        min_val_loss = 100
        for epoch in range(self.EPOCHS):
            loss_epoch = 0
            loss = 0        
            for idx, (image,label) in enumerate(self.train_dataloader): # _는 들어오는거 무시하는 것
                optimizer.zero_grad()
                if self.use_cuda:
                    image = image.to(device)
                    label = label.to(device)
                denoised_image = model(image)
                loss = criterion(denoised_image,label)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
            train_loss_avg = loss_epoch / len(self.train_dataloader)
            train_loss.append(train_loss_avg)
            
            loss_epoch = 0
            loss_valid = 0
            
            for idx, (image, label) in enumerate(self.val_dataloader):
                with torch.no_grad():
                    if self.use_cuda:
                        image = image.to(device)
                        label = label.to(device)
                    denoised_image = model(image)
                    loss_valid = criterion(denoised_image,label)
                    loss_epoch += loss_valid.item()
                    
            val_loss_avg = loss_epoch / len(self.val_dataloader)        
            val_loss.append(val_loss_avg)
            epochs.append(epoch+1)
            print("Epoch: " + str(epoch+1) + " train_loss: " + str(train_loss_avg) + " val_loss: " + str(val_loss_avg))
                    
            if val_loss_avg < min_val_loss:
                min_val_loss = val_loss_avg
                torch.save({
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch':epoch+1
                            }, self.CH_DIR + '/chk_9_Aug_lr_0.001.pt')
                print('Saving Model...')
            
        plt.plot(epochs, train_loss, label="train loss", color="red",linestyle=':')
        plt.plot(epochs, val_loss, label="val loss", color="green", linestyle=':')
        plt.title("Loss Curve")
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()
        plt.savefig('Loss Curve.png')
            
            
class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def mseloss(self, image, target): # 편차 제곱의 평균
        x = (image - target)**2 
        return torch.mean(x)
        
    def forward(self, image, target):
        return self.mseloss(image, target)
    

if __name__ == '__main__': # 이 py파일을 main으로 쓸때만 실행
    TrainUnet = Train()
    TrainUnet.train()
                