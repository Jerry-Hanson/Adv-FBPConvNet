import torch 
from ignite.metrics import SSIM,PSNR


def mse_(img1 : torch.tensor, img2 : torch.tensor):
    batch_size, c, w, h = img1.shape
    return ((img1 - img2) ** 2).reshape((-1, c * w * h)).mean(axis = 1) 

def batch_mse(img1 : torch.tensor, img2 : torch.tensor):
    return mse_(img1, img2).detach().cpu().tolist() 


def batch_rmse(img1 : torch.tensor, img2 : torch.tensor):
    return torch.square(mse_(img1, img2)).detach().cpu().tolist() 

def batch_psnr2(img1 : torch.tensor, img2 : torch.tensor, data_range : float, device : str = "cpu"):  
    # 计算每一张图片的ssim 
    ssim = PSNR(data_range = data_range, device = device)
    ssim_list = []
    for i in range(img1.shape[0]):
        ssim.reset()
        ssim.update((img1[i].unsqueeze(0), img2[i].unsqueeze(0)))
        
        ssim_list.append(ssim.compute()) 
    return ssim_list 

def batch_ssim(img1 : torch.tensor, img2 : torch.tensor, data_range : float, device : str = "cpu"):  
    # 计算每一张图片的ssim 
    ssim = SSIM(data_range = data_range, device = device)
    ssim_list = []
    for i in range(img1.shape[0]):
        ssim.reset()
        ssim.update((img1[i].unsqueeze(0), img2[i].unsqueeze(0)))
        
        ssim_list.append(ssim.compute()) 
    return ssim_list 


def batch_psnr(img1 : torch.tensor, img2 : torch.tensor, data_range : float):
    return 10 * torch.log10((data_range ** 2) / mse_(img1, img2)).detach().cpu().tolist() 


if __name__ == "__main__":
    img1 = torch.zeros(64, 1, 512, 512)
    img2 = torch.ones(64, 1, 512, 512) + torch.randn(img1.shape) 
    img3 = torch.zeros(2, 1, 512, 512)
    print(batch_psnr2(img1, img2, data_range = 256)) 