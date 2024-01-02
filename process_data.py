from tools.CTLib import create_sinogram_fan, recon_fan
import os 
import numpy as np 


if __name__ == "__main__":
    process_path_list = ["/data/DATA/yaoqiulei/mayo2016/processed/train/full_3mm", "/data/DATA/yaoqiulei/mayo2016/processed/test/full_3mm"]
    mode_list = ["train", 'test']

    for process_dir, mode in zip(process_path_list, mode_list):
        for file_name in os.listdir(process_dir):
            file_name_prefix = file_name.split(".")[0]
            
            file_path = os.path.join(process_dir, file_name)
            print(f"processing {file_path}")
            full_view_img = np.load(file_path)

            for view_number in [180, 144, 72]:
                sinogram = create_sinogram_fan(full_view_img, angles = np.linspace(0, 360, view_number, False) * (np.pi / 180))
                # 将sinogram保存到指定的位置
                np.save(os.path.join(f"/data/DATA/yaoqiulei/mayo2016/sinogram/{mode}/{str(view_number)}-view/", f"{file_name_prefix}.npy") , sinogram)
                
                # 重建图片
                recon_image = recon_fan("FBP_CUDA", sinogram, full_view_img.shape, np.linspace(0, 360, view_number, False) * (np.pi / 180))
                # 将重建后的图片保存到指定位置
                np.save(os.path.join(f"/data/DATA/yaoqiulei/mayo2016/processed/{mode}/{str(view_number)}-view/", f"{file_name_prefix}.npy"), recon_image)
                