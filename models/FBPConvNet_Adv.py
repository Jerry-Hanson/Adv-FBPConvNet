import torch
import torch.nn as nn
from .network import UnetGenerator, init_weights
from .FBPConvNet import FBPCONVNet
    
class FBPCONVNet_Adv(nn.Module):
    def __init__(self, lr = 0.001, lambda1 = 1, lambda2 = 1, epsilon = 0.1):
        """
        lambda1, lambda2 represent the coefficients in loss function 
        epsilon represents the level of noise
        """
        super(FBPCONVNet_Adv, self).__init__()
        self.lambda1 = lambda1 
        self.lambda2 = lambda2
        self.epsilon = epsilon

        
        # create network model
        self.backbone = FBPCONVNet()
        self.noise_module = UnetGenerator(input_nc = 1, output_nc = 1, num_downs = 7)
        init_weights(self.noise_module)
        self.optimizer_backbone = torch.optim.Adam(self.backbone.parameters(), lr = lr)
        self.optimizer_noise_module = torch.optim.Adam(self.noise_module.parameters(), lr = lr)

    def set_input(self, input_X, label):
        # 随机选择50%的数据做perturbation
        batch_size, _, _, _ = input_X.shape
        perturbation_size = batch_size // 2
        random_indices = torch.randperm(batch_size)
        perturbation_indices = random_indices[:perturbation_size]
        clean_indices = random_indices[perturbation_size:]

        self.perturbation_X = input_X[perturbation_indices]
        self.perturbation_label = label[perturbation_indices]
        self.clean_X = input_X[clean_indices]
        self.clean_label = label[clean_indices]



    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def optimize_parameters(self):
        self.optimizer_backbone.zero_grad()
        self.optimizer_noise_module.zero_grad() 


        # 首先更新FBPConvNet
        self.set_requires_grad(self.noise_module, False)
        self.set_requires_grad(self.backbone, True)
        self.noise = self.noise_module(self.clean_X)
        self.perturbation_output = self.backbone(self.noise + self.perturbation_X) 
        self.clean_output = self.backbone(self.clean_X)
        
        self.recon_loss_1 = nn.MSELoss()(self.clean_output, self.clean_label)
        self.recon_loss_2 = nn.MSELoss()(self.perturbation_output, self.perturbation_label)
        self.recon_loss = self.recon_loss_1 + self.recon_loss_2
        self.recon_loss.backward() 
        self.optimizer_backbone.step()

        # 更新noise_module
        self.set_requires_grad(self.noise_module, True)
        self.set_requires_grad(self.backbone, False)
        self.noise = self.noise_module(self.perturbation_X)
        self.perturbation_output = self.backbone(self.perturbation_X + self.noise) 
        

        batch_size, _, _, _ = self.noise.shape
        self.G_A_loss = - 1 * self.lambda1 * nn.MSELoss()(self.perturbation_output, self.perturbation_label)
        self.G_B_loss = self.lambda2 * torch.clip(((self.noise.reshape(batch_size, -1) ** 2).mean() - self.epsilon), min = 0).sum()

        self.G_loss = self.G_A_loss + self.G_B_loss
        self.G_loss.backward() 
        self.optimizer_noise_module.step() 
        
    def forward(self, x):
        return self.backbone(x)


        

    

if __name__ == "__main__":
    device = 'cuda'
    model = FBPCONVNet_Adv()
    # model.load_state_dict(torch.load("/home/yaoqiulei/RobustMLProject/ckpt/FBPConvNetAdv/FBPCONVNet_Adv_FBPConvAdv-180view-1-1-0.0001_2023-12-10-01-17/200_2023-12-10-07-11.pt"))
    model = model.to(device) 
    X = torch.randn((8, 1, 512, 512))
    X = X.to(device) 
    print(model(X).shape) 