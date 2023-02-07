import torch.utils.data as data
import scipy.io as io
import numpy as np
import torch
from UNet import UNet
import argparse
from torchvision.utils import save_image
from train import Train
from torchvision import transforms


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
    
    
class Test():
    def __init__(self):
        self.STD, self.DATA_DIR, self.CHECKPOINT = self.__get_args__()
        # self.transforms = Train().transforms
        
    def __get_args__(self):
        parser = argparse.ArgumentParser(description='Parameters')
        parser.add_argument('--std', type=float, default=0.1)
        parser.add_argument('--data_dir', type=str, default='./data/test')
        parser.add_argument('--checkpoint', type=str,
                            default='./checkpoints/chk_9_Aug_lr_0.001.pt') # TODO: 원하는 checkpoint 이름으로 수정
        args = parser.parse_args()
        return args.std, args.data_dir, args.checkpoint

    def img_denorm(self, img, mean, std): # Denormalize: Normalize를 이전의 상태로
        denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))
        res = denormalize(img)
        res = torch.clip(res, 0, 1)
        return(res)
    
    def make_dataset_X_noise_power(self, h5_path, noise_power):
        f = io.loadmat(h5_path)
        img_all = None
        
        if img_all is None:
            img_all = f.get('img_all')

        # img_all.shape = [96,160,9,21,19]
        # print(img_all.shape)
        
        X = np.complex64(img_all[:,:,0,:,:])
        
        X = torch.tensor(X)
        
        X = X.permute(2,3,0,1)
        
        X = torch.sqrt(torch.pow(X.real,2) + torch.pow(X.imag,2))

        std = 0.01 * noise_power
        noise = std * torch.randn(X.shape)
        for i in range(X.shape[1]):
            noise[:,i,:,:] = noise[:,i,:,:] * torch.std(X[:,i,:,:])
        X = X + noise

        print("val train data size:")
        print(X.shape)
        return (
            X.float()
        )
        
    def test(self):
        # img_recon_cor_EX5559_p106.mat
        # img_recon_cor_EX5588_p111.mat
        print("#############")
        print("X")
        
        X = self.make_dataset_X_noise_power('/SSD5_8TB/LJH/DWI_data/test/img_recon_cor_EX5588_p111.mat',20)        
        
        # LLR_cor_EX5559_p106.mat
        # LLR_cor_EX5588_p111.mat
        print("#############")
        print("Y")
        f = io.loadmat('/SSD5_8TB/LJH/DWI_data/test_LLR/LLR_cor_EX5588_p111.mat')
        img_all = None
        
        if img_all is None:
            img_all = f.get('denoised_img')

        # img_all.shape = [96,160,21,19]
        print(img_all.shape)
        Y = np.complex64(img_all)
        Y = torch.tensor(Y)
        Y = Y.permute(2,3,0,1)
        Y = torch.sqrt(torch.pow(Y.real,2) + torch.pow(Y.imag,2))
        print(Y.shape) 
        print("################")
        X = (X-torch.min(X))/(torch.max(X)-torch.min(X))*2
        Y = (Y-torch.min(Y))/(torch.max(Y)-torch.min(Y))*2  
        test_dataset = CombinedDataset(X.float(), Y.float())
        
        # testset = ImageDataset(self.DATA_DIR, transform=self.transforms)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=21)
        
        checkpoint = torch.load(self.CHECKPOINT, map_location=torch.device('cpu'))
        model_test = UNet(in_channels=19, out_channels=19).float()
        model_test.load_state_dict(checkpoint['model_state_dict'])
        model_test = model_test.cpu()
        model_test.train()
            
        for idx, (image,label) in enumerate(testloader):
            print("Data Loader -> image, label")
            print(image.shape) # torch.Size([batch_size, 19, 96, 160])
            print(label.shape) # torch.Size([batch_size, 19, 96, 160])

            ### 이미지 저장
            for i in range((image.cpu()).size()[1]): # batch 개수 파악
                ## 1.원본 이미지 저장
                # print("origin max: " + str(image.max()))
                # print("origin min: " + str(image.min()))
                save_image(label[8,i,:,:],'./results/chk_9/original_image_batch_'+str(idx)+"_"+str(i)+'.jpg')
          
                ## 2.노이즈 이미지 저장
                noisy_image = image.cpu()[8,i,:,:]
                # print("noisy_image max: " + str(noisy_image.max()))
                # print("noisy_image min: " + str(noisy_image.min()))
                save_image(noisy_image,'./results/chk_9/noisy_image_batch_'+str(idx)+"_"+str(i)+'.jpg')
                
                ## 3.디노이즈 이미지 저장
                denoisy_image = model_test(image.cpu())
                # print("denoisy_image max: " + str(denoisy_image.max()))
                # print("denoisy_image min: " + str(denoisy_image.min()))
                save_image(denoisy_image[8,i,:,:],'./results/chk_9/denoised_image_batch_'+str(idx)+"_"+str(i)+'.jpg')
            print("##########################################################")    
            
            label = label.detach().numpy()
            image = image.detach().numpy()
            denoisy_image = denoisy_image.detach().numpy()
            print(label.shape)
            print(image.shape)
            print(denoisy_image.shape)
            io.savemat("label20.mat",{"label":label})
            io.savemat("input20.mat",{"input":image})
            io.savemat("output20.mat",{"output":denoisy_image})
            print("Batch " + str(idx) + " Done...")
        print('End')
    
if __name__ == '__main__': # 이 py파일을 main으로 쓸때만 실행
    TestUnet = Test()
    TestUnet.test()