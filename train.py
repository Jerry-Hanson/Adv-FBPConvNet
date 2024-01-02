import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 
from tools.dataset import Mayo2016Dataset
from tools.config_parser import ConfigParser
from models.FBPConvNet import FBPCONVNet
import torch.optim as optim 
import matplotlib.pyplot as plt 
from tools.metric import batch_rmse, batch_psnr, batch_ssim, batch_psnr2
import wandb 
from tools.model_handler import model_summary, model_profile 
from pprint import pprint 
import numpy as np 
import os 
from tools.base import get_current_time, setup_seed
import time 



def train_one_epoch(train_dataloader, model, optimizer, criterion, device, use_wandb):
    model.train() 
    batch_loss_list = []

    for i, (X, y) in enumerate(train_dataloader):
        X = X.to(device) 
        y = y.to(device)    
        
        
        output = model(X)
        loss = criterion(output, y)
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 

        loss_value = loss.cpu().detach().tolist() 

        # log loss value 
        if use_wandb:
            wandb.log({"train_batch_loss":loss_value})
        
        batch_loss_list.append(loss_value)
        
    return batch_loss_list 


def test_one_epoch(test_dataloader, model, data_range, device, use_wandb):
    model.eval() 
    ssim_list = []
    psnr_list = []
    rmse_list = []
    

    with torch.no_grad():
        for i, (X, y) in enumerate(test_dataloader):
            X = X.to(device) 
            y = y.to(device) 

            output = model(X)
                
         
            ssim_list.extend(batch_ssim(output, y, data_range = data_range))
            psnr_list.extend(batch_psnr2(output, y, data_range = data_range))
            rmse_list.extend(batch_rmse(output, y))

    if use_wandb:
        wandb.log({"test_ssim(mean)":np.mean(ssim_list), "test_rmse(mean)":np.mean(rmse_list), "test_psnr(mean)":np.mean(psnr_list)})

    return ssim_list, psnr_list, rmse_list   

def main(args):
    setup_seed()

    train_dataset = Mayo2016Dataset(X_folder_path = args.train_X_folder_path, y_folder_path = args.train_y_folder_path) 
    test_dataset = Mayo2016Dataset(X_folder_path = args.test_X_folder_path, y_folder_path = args.test_y_folder_path)
    
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size,
                                  shuffle = True, num_workers = args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, 
                                 shuffle = False, num_workers=args.num_workers)

    ## load model 
    model = FBPCONVNet() 
    model = model.to(args.device)

    # compute the size of model 
    dummy_input = torch.randn(args.batch_size, 1, 512, 512).to(args.device)
    flops, params = model_profile(model, dummy_input)
    del dummy_input  # 删除变量，防止占用内存和显存

    ## load train settings
    # optimizer 
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    criterion = nn.MSELoss()

    config = {
                "lr":args.lr,
                "model":model._get_name(), 
                "dataset":train_dataset.__class__.__name__, 
                "epoch":args.epoch, 
                "weight_decay" : args.weight_decay, 
                "device":args.device, 
                "data_range" : args.data_range, 
                "num_workers" : args.num_workers,
                "flops/G" : flops, 
                "params/M" : params
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
                        optimizer = optimizer, 
                        criterion = criterion,
                        device = args.device, 
                        use_wandb = args.use_wandb)
        
        time_stamp_2 = time.time()
        ssim_list, psnr_list, rmse_list = test_one_epoch(test_dataloader = test_dataloader, 
                                                        model = model, 
                                                        data_range = args.data_range, 
                                                        device = args.device, 
                                                        use_wandb = args.use_wandb)
        
        time_stamp_3 = time.time() 

        if i % args.log_print_interval == 0:
            pprint(f"--> epoch:{i}, train_epoch_loss(mean):{np.mean(train_batch_loss_list)} \
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
            for train_batch_loss in train_batch_loss_list:
                f.write(str(train_batch_loss) + "\n")

        with open(os.path.join(log_save_folder, "test_metrics.txt"), "a") as f:
            f.write(f"{str(np.mean(ssim_list))},{str(np.mean(psnr_list))},{str(np.mean(rmse_list))}" + "\n")

    wandb.finish()


if __name__ == "__main__":
    argparser = ConfigParser() 
    pprint(argparser.cfg)
    main(argparser.cfg)