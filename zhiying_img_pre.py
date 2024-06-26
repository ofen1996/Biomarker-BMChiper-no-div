import os
import zarr
from PIL import Image
import tifffile
import numpy as np
from numcodecs import Blosc


def read_base_msg(pics_dir):
    pic_names = os.listdir(pics_dir)

    channel = set()
    row = set()
    colum = set()
    pic_paths = []

    for pic_name in pic_names:
        pic_paths.append(os.path.join(pics_dir, pic_name))
        pic_name = pic_name.split(";")
        channel.add(pic_name[0][1:])
        row.add(pic_name[2].split(",")[1])
        colum.add(pic_name[2].split(",")[0])

    image_path = pic_paths[0]
    with Image.open(image_path) as img:
        shape = img.size
    return len(row), len(colum), len(channel), shape[::-1], pic_paths


if __name__ == '__main__':

    pics_dir = r"E:\test\志盈\图像"
    save_dir, img_name = os.path.split(pics_dir)[0], os.path.split(pics_dir)[1]
    save_path = os.path.join(save_dir, img_name+"-zarr")

    row, colum, channel, shape, pic_paths = read_base_msg(pics_dir)

    # # 开始保存
    compressor = Blosc(cname='zstd', clevel=8, shuffle=Blosc.BITSHUFFLE)
    img_zarr = zarr.open(save_path, mode="w", shape=(shape[0]*row, shape[1]*colum, 3), chunks=shape, dtype="uint8")
    for pic_path in pic_paths:
        pic_name = os.path.split(pic_path)[-1]
        pic_name = pic_name.split(";")
        channel = int(pic_name[0][1:])
        r = int(pic_name[2].split(",")[1])
        c = int(pic_name[2].split(",")[0])

        with Image.open(pic_path) as img:
            img_zarr[shape[0]*r:shape[0]*(r+1), shape[1]*c:shape[1]*(c+1), channel] = np.asarray(img.convert("L"))
    # z1 = zarr.zeros((10000, 10000), chunk=(2448, 2048),dtype="uint8")
    # zarr.save("E://temp.zarr", z1)