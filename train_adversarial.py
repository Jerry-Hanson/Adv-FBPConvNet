import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 
from tools.dataset import Mayo2016Dataset
from tools.config_parser import ConfigParser
from models.FBPConvNet_Adv import FBPCONVNet_Adv
import torch.optim as optim 
import matplotlib.pyplot as plt 
from tools.metric import batch_rmse, batch_psnr, batch_ssim, batch_psnr2
import wandb 
from tools.model_handler import model_profile 
from pprint import pprint 
import numpy as np 
import os 
from tools.base import get_current_time, setup_seed
import time 
import tifffile



def train_one_epoch(train_dataloader, model, device):
    model.train() 
    batch_loss_list = []

    for i, (X, y) in enumerate(train_dataloader):
        X = X.to(device) 
        y = y.to(device) 
        
        model.set_input(X, y)
        model.optimize_parameters()

        batch_loss_list.append({
            "n_iter": i,
            "G_A_loss": model.G_A_loss.cpu().detach().tolist(), 
            "G_B_loss": model.G_B_loss.cpu().detach().tolist(), 
            "G_loss": model.G_loss.cpu().detach().tolist(),
            "recon_loss_1": model.recon_loss_1.cpu().detach().tolist(), 
            "recon_loss_2": model.recon_loss_2.cpu().detach().tolist(), 
            "recon_loss": model.recon_loss.cpu().detach().tolist()
        })

        
    return batch_loss_list 


def test_one_epoch(test_dataloader, model, data_range, device):
    model.eval() 
    ssim_list = []
    psnr_list = []
    rmse_list = []
    

    with torch.no_grad():
        for i, (X, y) in enumerate(test_dataloader):
            X = X.to(device) 
            y = y.to(device) 

            output = model.backbone(X)  # 这里的backbone是一个FBPConvNet
            
         
            ssim_list.extend(batch_ssim(output, y, data_range = data_range))
            psnr_list.extend(batch_psnr2(output, y, data_range = data_range))
            rmse_list.extend(batch_rmse(output, y))

    return ssim_list, psnr_list, rmse_list   

def main(args):
    setup_seed()

    train_dataset = Mayo2016Dataset(X_folder_path = args.train_X_folder_path, y_folder_path = args.train_y_folder_path) 
    test_dataset = Mayo2016Dataset(X_folder_path = args.test_X_folder_path, y_folder_path = args.test_y_folder_path)
    
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size,
                                  shuffle = True, num_workers = args.num_workers, 
                                  drop_last = True)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, 
                                 shuffle = False, num_workers=args.num_workers, 
                                 drop_last = False)

    ## load model 
    model = FBPCONVNet_Adv(lr = args.lr, lambda1 = args.lambda1, lambda2 = args.lambda2, epsilon = args.epsilon) 
    model = model.to(args.device)

    ## load train settings
    config = {
                "lr":args.lr,
                "model":model._get_name(), 
                "dataset":train_dataset.__class__.__name__, 
                "epoch":args.epoch, 
                "weight_decay" : args.weight_decay, 
                "device":args.device, 
                "data_range" : args.data_range, 
                "num_workers" : args.num_workers,
            }

    # set log save folder
    log_save_folder = os.path.join(args.log_save_folder, f"{args.dataset_name}_{model._get_name()}_{str(args.version)}_{get_current_time()}") 
    os.makedirs(log_save_folder, exist_ok = True)
    
    # set ckpt save folder 
    ckpt_save_folder = os.path.join(args.ckpt_save_folder, f"{args.dataset_name}_{model._get_name()}_{str(args.version)}_{get_current_time()}")
    os.makedirs(ckpt_save_folder, exist_ok = True)

    # log
    pprint(config)

    # init wandb 
    if args.use_wandb:
        table_name = f"{model._get_name()}_{str(args.version)}_{get_current_time()}"
        wandb.init(project=args.wandb_project, config = config, name = table_name)    
    

    for i in range(1, args.epoch + 1):
        time_stamp_1 = time.time()
        train_batch_loss_list = train_one_epoch(train_dataloader = train_dataloader, 
                        model = model,
                        device = args.device)
        
        G_A_loss_values = [entry["G_A_loss"] for entry in train_batch_loss_list]
        G_A_loss_mean = np.mean(G_A_loss_values)
        
        G_B_loss_values = [entry["G_B_loss"] for entry in train_batch_loss_list]
        G_B_loss_mean = np.mean(G_B_loss_values)
        
        G_loss_values = [entry["G_loss"] for entry in train_batch_loss_list]
        G_loss_mean = np.mean(G_loss_values)
        
        recon_loss_1_values = [entry["recon_loss_1"] for entry in train_batch_loss_list]
        recon_loss_1_mean = np.mean(recon_loss_1_values)

        recon_loss_2_values = [entry["recon_loss_2"] for entry in train_batch_loss_list]
        recon_loss_2_mean = np.mean(recon_loss_2_values)

        recon_loss_values = [entry["recon_loss"] for entry in train_batch_loss_list]
        recon_loss_mean = np.mean(recon_loss_values)


        time_stamp_2 = time.time()
        ssim_list, psnr_list, rmse_list = test_one_epoch(test_dataloader = test_dataloader, 
                                                        model = model, 
                                                        data_range = args.data_range, 
                                                        device = args.device)
        tifffile.imwrite(f"./images/{str(i)}.tif", model.noise.cpu().detach().numpy()[0][0].astype(np.float32)) 

        time_stamp_3 = time.time() 

        if i % args.log_print_interval == 0:
            pprint(f"--> epoch:{i}, G_A_loss(mean):{G_A_loss_mean} \
                   G_B_loss(mean):{G_B_loss_mean}\
                   G_loss(mean):{G_loss_mean}\
                   recon_loss_1(mean):{recon_loss_1_mean}\
                   recon_loss_2(mean):{recon_loss_2_mean}\
                   recon_loss(mean):{recon_loss_mean}\
                   test_ssim(mean):{np.mean(ssim_list)} test_psnr(mean):{np.mean(psnr_list)}\
                    test_rmse(mean):{np.mean(rmse_list)}\
                    training time:{time_stamp_2 - time_stamp_1}s\
                    testing time:{time_stamp_3 - time_stamp_2}s")

        if i % args.ckpt_save_interval == 0:
            # save model checkpoint 
            save_path = os.path.join(ckpt_save_folder, str(i) + "_" + get_current_time() + ".pt")
            torch.save(model.state_dict(), save_path) 
            pprint(f"checkpoint is saved to : {save_path}")


        # 将训练过程中产生的数据持久化存储到文件中
        with open(os.path.join(log_save_folder, "train_loss_batch.txt"), "a") as f:
        # G_A_loss, G_B_loss, G_loss, recon_loss_1, recon_loss_2, recon_loss
            for train_batch_loss in train_batch_loss_list:
                f.write(f"{train_batch_loss['G_A_loss']},{train_batch_loss['G_B_loss']},{train_batch_loss['G_loss']},{train_batch_loss['recon_loss_1']},{train_batch_loss['recon_loss_2']},{train_batch_loss['recon_loss']}\n")

        with open(os.path.join(log_save_folder, "test_metrics.txt"), "a") as f:
            f.write(f"{str(np.mean(ssim_list))},{str(np.mean(psnr_list))},{str(np.mean(rmse_list))}" + "\n")



if __name__ == "__main__":
    argparser = ConfigParser() 
    pprint(argparser.cfg)
    main(argparser.cfg)