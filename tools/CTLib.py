from .logtool import RED, ENDC
import matplotlib.pyplot as plt
import astra 

def create_sinogram_fan(image,
                        angles,
                        vol_geom_size=None,
                        dect_w=1,
                        source_ori=1000,
                        ori_detector=500,
                        dect_count=1024):
    if vol_geom_size is None:
        vol_geom_size = image.shape

    vol_geom_fan = astra.create_vol_geom(vol_geom_size)
    proj_geom_fan = astra.create_proj_geom('fanflat', dect_w, dect_count,
                                           angles,
                                           source_ori, ori_detector)

    proj_fan_id = astra.create_projector('cuda', proj_geom_fan, vol_geom_fan)
    sino_fan_id, sino_gram = astra.create_sino(image, proj_fan_id)

    astra.projector.delete(proj_fan_id)
    astra.projector.delete(sino_fan_id)
    astra.clear()
    return sino_gram


def recon_fan(alg, sino, vol_geom_size, angles, filter="ram-lak", source_ori=1000, ori_detector=500, dect_w=1,
              iterations=-1):
    astra.algorithm.clear()
    vol_geom_fan = astra.create_vol_geom(vol_geom_size)
    proj_geom_fan = astra.create_proj_geom('fanflat', dect_w, sino.shape[1],
                                           angles,
                                           source_ori, ori_detector)
    proj_fan_id = astra.create_projector('cuda', proj_geom_fan, vol_geom_fan)
    sinogram_fan_id = astra.data2d.create('-sino', proj_geom_fan, sino)
    rec_fan_id = astra.data2d.create('-vol', vol_geom_fan)
    cfg_fan = astra.astra_dict(alg)
    cfg_fan['ReconstructionDataId'] = rec_fan_id
    cfg_fan['ProjectionDataId'] = sinogram_fan_id

    cfg_fan['option'] = {'ShortScan': False}

    if alg.startswith("FBP"):
        cfg_fan["FilterType"] = filter

    ## cfg_fan['ProjectorId'] = proj_fan_id
    alg_fan_id = astra.algorithm.create(cfg_fan)

    # iterative algorithm
    if iterations != -1:
        astra.algorithm.run(alg_fan_id, iterations)
    else:
        astra.algorithm.run(alg_fan_id)

    rec_fan = astra.data2d.get(rec_fan_id)
    astra.algorithm.delete(alg_fan_id)
    astra.data2d.delete(rec_fan_id)
    astra.data2d.delete(sinogram_fan_id)
    astra.projector.delete(proj_fan_id)
    # rec_fan[rec_fan < 0] = 0
    # rec_fan = np.flipud(rec_fan)
    astra.algorithm.clear()
    astra.clear()
    return rec_fan

if __name__ == "__main__":
    import numpy as np 
    X = np.load("/data/DATA/yaoqiulei/mayo2016/processed/train/full_3mm/L067_206_target.npy")
    # 首先将图片前投影到sinogram
    sinogram = create_sinogram_fan(X, angles = np.linspace(0, 360, 180) * (np.pi / 180), source_ori = 1000, ori_detector = 500, dect_count = 1024)
    print(sinogram.shape)
    plt.imsave("./sinogram.png", sinogram, cmap = 'gray')
    recon_image = recon_fan(alg = "FBP_CUDA", sino = sinogram, vol_geom_size = X.shape, angles = np.linspace(0, 360, 180) * (np.pi / 180))
    print(recon_image.shape)
    plt.imsave("./recon_img.png", recon_image, cmap = 'gray')
