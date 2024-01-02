from datetime import datetime 
import torch
import os
import numpy as np
import random


def get_current_time(str_format = r"%Y-%m-%d-%H-%M"):
    return datetime.now().strftime(str_format)


def setup_seed(seed=0):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
