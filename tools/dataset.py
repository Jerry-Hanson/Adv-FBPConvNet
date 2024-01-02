from torch.utils.data import Dataset
import numpy as np 
import os 
import torch 


class Mayo2016Dataset(Dataset):
    def __init__(self, X_folder_path, y_folder_path):
        super().__init__()
        self.X_file_path_list = []
        self.y_file_path_list = []

        for file_name in os.listdir(X_folder_path):
            name = "_".join(file_name.split("_")[:2])
            self.X_file_path_list.append(os.path.join(X_folder_path, file_name))
            # y_file_name = name + "_target.npy"
            y_file_name = file_name
            # 检查paired data是否存在
            if not os.path.exists(os.path.join(y_folder_path, y_file_name)):
                raise Exception(f"target file {os.path.join(y_folder_path, y_file_name)} does not exist!!!")
            
            self.y_file_path_list.append(os.path.join(y_folder_path, y_file_name))

            

    def __len__(self):
        return len(self.X_file_path_list)
    
    def __getitem__(self, idx):
        input_X = np.load(self.X_file_path_list[idx])
        target_y = np.load(self.y_file_path_list[idx])
        
        # convert data from np.ndarray to torch.tensor
        input_X = torch.tensor(input_X, dtype = torch.float32).unsqueeze(dim = 0)
        target_y = torch.tensor(target_y, dtype = torch.float32).unsqueeze(dim = 0)
        
        return input_X, target_y
    
# if __name__ == "__main__":
#     dataset = Mayo2016Dataset(X_folder_path = "/data/DATA/yaoqiulei/mayo2016/processed/quarter_3mm", y_folder_path="/data/DATA/yaoqiulei/mayo2016/processed/full_3mm")
#     print(dataset[0][0].shape)
