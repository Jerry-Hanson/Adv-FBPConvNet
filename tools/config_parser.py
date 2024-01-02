from omegaconf import OmegaConf
import os, argparse, random

# 为每一次训练任务创建一个config类
# 读取某一个文件夹中的所有yaml文件
class ConfigParser:
    def __init__(self):
        self.cfg = OmegaConf.create()
        
        # 解析命令行参数
        cli_args = self.parse_args()

        # 整合命令行参数和配置文件参数, 命令行参数的优先级最大
        self.merge_config_folder(cli_args)

        self.merge_config_file(cli_args.base_cfg)

        
    def parse_args(self): 
        """
        解析命令行参数
        主要模型配置、训练的配置主要在配置文件中
        """
        parser = argparse.ArgumentParser(description="Cli Args")
        parser.add_argument("--mode", default = "train", type = str)
        parser.add_argument("--cfg_dir", default = "/home/yaoqiulei/RobustMLProject/config/FBPConvAdv", type = str)
        parser.add_argument("--base_cfg", default = "/home/yaoqiulei/RobustMLProject/config/base-180-view.yaml", type = str)
        parser.add_argument("--device", default = "cuda", type = str)
        parser.add_argument("--save_path", default = "./output/", type = str)

        ## 其他客制化需求
        args = parser.parse_known_args()[0]
        return args
    
    def merge_config_folder(self, cli_args = None):
        """
        整合yaml、命令行的参数
        Args:
            cli_args: 命令行参数
        Return: None
        """
        # 遍历所有的yaml文件
        files_dict = self.search_files(cli_args.cfg_dir, file_format = "yaml")


        # 加载任务的配置文件
        for name, path in files_dict.items():
            print(f"loading config file : {path}")
            self.cfg.merge_with(OmegaConf.load(path))
        
        # 整合命令行配置
        if cli_args is not None:
            cli_args = [f"{name}={value}" for name, value in vars(cli_args).items()]
            cli_conf = OmegaConf.from_cli(cli_args)
            self.cfg.merge_with(cli_conf)


    def merge_config_file(self, file_path):
        """
        向cfg文件中添加新的配置文件信息
        """
        print(f"loading config file : {file_path}")
        self.cfg.merge_with(OmegaConf.load(file_path))

    def search_files(self, file_dir, file_format):
        """
        搜索file_dir目录下的所有file_format的文件
        Args:
            file_dir: 需要搜寻的目录
            file_format: 

        Returns: 所有以file_format为格式的文件路径, 返回数据是字典，以文件名为key，文件路径为value
        """
        file_list, file_dict = [], {}
        for path, dirs, files in os.walk(file_dir):
            [file_list.append(os.path.join(path, file)) for file in files if file.split(".")[-1] == file_format] 
        
        if len(file_list) == 0:
            return file_dict
        
        if '\\' in file_list[0]:
            return {file_path.split('.')[-2].split('\\')[-1]:file_path for file_path in file_list} 
        elif '/' in file_list[0]:
            return {file_path.split('.')[-2].split('/')[-1]:file_path for file_path in file_list}  


if __name__ == "__main__":
    # 加载一个文件夹中的配置文件
    path = ConfigParser().cfg
    print(path)
    



