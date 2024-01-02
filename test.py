import torch 
from models.FBPConvNet import FBPCONVNet
from models.FBPConvNet_Adv import FBPCONVNet_Adv
import numpy as np 
import matplotlib.pyplot as plt
import tifffile as tiff 
import os 


def process_one_image(model, data, device):
    # data.shape = [512, 512]
    model.eval()
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    data = data.to(device) 
    data = data.unsqueeze(0).unsqueeze(0)
    model = model.to(device) 
    
    ret = model(data)
    output = ret[0][0].cpu().detach().numpy() 
    
    # output = model(data)[0][0].cpu().detach().numpy()

    return output



def process_batch_image(model, data, device):
    # data.shape = [b, 1, 512, 512]
    model.eval() 
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    data = data.to(device) 
    model = model.to(device) 
    output = model(data).cpu().detach().numpy() 

    return output

if __name__ == "__main__":
    # model = FBPCONVNet_Adv()
    model = FBPCONVNet()
    output_folder = "/home/yaoqiulei/RobustMLProject/results/FBPConvNet"
    state_dict = torch.load("/data/DATA/yaoqiulei/RobustMLProject/FBPConvNet/ckpt/mayo2016-180view_FBPCONVNet_1_2024-01-01-22-31/100_2024-01-01-23-02.pt")
    # 因为在训练的时候使用了thop来计算模型的参数量和ops，需要移除这两个key
    state_dict.pop("total_ops")
    state_dict.pop("total_params")
    
    # input_list = [
    #     {"input_path":"/data/DATA/yaoqiulei/mayo2016/processed/test/180-view/L506_0_target.npy", "label_path" : "/data/DATA/yaoqiulei/mayo2016/processed/test/full_3mm/L506_0_target.npy"},  
    #     {"input_path":"/data/DATA/yaoqiulei/mayo2016/processed/test/180-view/L506_40_target.npy", "label_path" : "/data/DATA/yaoqiulei/mayo2016/processed/test/full_3mm/L506_40_target.npy"},  
    #     {"input_path":"/data/DATA/yaoqiulei/mayo2016/processed/test/180-view/L506_80_target.npy", "label_path" : "/data/DATA/yaoqiulei/mayo2016/processed/test/full_3mm/L506_80_target.npy"},  
    #     {"input_path":"/data/DATA/yaoqiulei/mayo2016/processed/test/180-view/L506_120_target.npy", "label_path" : "/data/DATA/yaoqiulei/mayo2016/processed/test/full_3mm/L506_120_target.npy"},  
    #     {"input_path":"/data/DATA/yaoqiulei/mayo2016/processed/test/180-view/L506_180_target.npy", "label_path" : "/data/DATA/yaoqiulei/mayo2016/processed/test/full_3mm/L506_180_target.npy"},  
    # ]
    input_list = []
    
    for i in range(200):
        input_path = f"/data/DATA/yaoqiulei/mayo2016/processed/test/180-view/L506_{i}_target.npy"
        label_path = f"/data/DATA/yaoqiulei/mayo2016/processed/test/full_3mm/L506_{i}_target.npy"
        input_list.append({"input_path":input_path, "label_path":label_path})


    model.load_state_dict(state_dict) 

    for input_ in input_list:
        name = input_['input_path'].split("/")[-1].split(".")[0]
        X = np.load(input_['input_path'])
        raw_image = np.load(input_['label_path'])
        
        output = process_one_image(model, X, 'cuda')
        
        # 将output, raw_image, difference_map保存到指定目录下
        tiff.imwrite(os.path.join(output_folder, name + "_FBPConvNet_100_output.tif"), output)
        tiff.imwrite(os.path.join(output_folder, name + "_raw_image.tif"), raw_image.astype(np.float32))
        tiff.imwrite(os.path.join(output_folder, name + "_FBPConvNet_100_difference.tif"), (raw_image - output).astype(np.float32))
