import argparse
import sys
import time

import numpy as np
from PIL import Image
import os
import tifffile
import tiffslide
import cv2
from need.ofen_tool import show_img

vipsbin = r'D:\work\python\vips-dev-8.15\bin'
os.environ['PATH'] = vipsbin + ';' + os.environ['PATH']
import pyvips
import openslide
## 海德星的小图拼接成大图


def save_pyramid_tif(im, save_path, compression="jpeg",  tile_width=512, tile_height=512):
    # openslide will add an alpha ... drop it
    if im.hasalpha():
        im = im[:-1]

    image_height = im.height
    image_bands = im.bands

    # split to separate image planes and stack vertically ready for OME
    im = pyvips.Image.arrayjoin(im.bandsplit(), across=1)

    # set minimal OME metadata
    # before we can modify an image (set metadata in this case), we must take a
    # private copy
    im = im.copy()
    im.set_type(pyvips.GValue.gint_type, "page-height", image_height)
    im.set_type(pyvips.GValue.gstr_type, "image-description",
                f"""<?xml version="1.0" encoding="UTF-8"?>
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
        <Image ID="Image:0">
            <!-- Minimum required fields about image dimensions -->
            <Pixels DimensionOrder="XYCZT"
                    ID="Pixels:0"
                    SizeC="{image_bands}"
                    SizeT="1"
                    SizeX="{im.width}"
                    SizeY="{image_height}"
                    SizeZ="1"
                    Type="uint8">
            </Pixels>
        </Image>
    </OME>""")

    im.tiffsave(save_path, compression=compression, tile=True,
                tile_width=tile_width, tile_height=tile_height,
                pyramid=True)


def generate_matrix(M, N):
    rows, cols = np.meshgrid(np.arange(1, M+1), np.arange(1, N+1), indexing='ij')
    matrix = np.stack((rows, cols), axis=-1)
    return matrix


def gen_HDX_tile_images(pics_dir, tiles_across, tiles_down):

    tiles = [pyvips.Image.new_from_file(
        os.path.join(pics_dir, f"IMG{str(y).zfill(3)}x{str(x).zfill(3)}.tif"))
        for y in range(1, tiles_down + 1) for x in range(1, tiles_across + 1)]
    whole_img = pyvips.Image.arrayjoin(tiles, across=tiles_across)
    # whole_img = whole_img.scRGB2BW(depth=30)  //二值化
    # whole_img = whole_img[0]
    whole_img = whole_img.rot90()  # 海德星图像需要转90度
    return whole_img[0]  #返回灰度


def HDX_image_prepare(pics_dir):
    img_names = os.listdir(pics_dir)

    rows = [int(name[3:6]) for name in img_names if name.endswith(".tif")]
    cols = [int(name[7:10]) for name in img_names if name.endswith(".tif")]
    # 设置矩阵形状
    row, col = max(rows), max(cols)
    shape_r_c = (row, col)

    tmp_im = pyvips.Image.new_from_file(os.path.join(pics_dir, img_names[0]))
    img_width, img_height = tmp_im.width, tmp_im.height
    return shape_r_c, (img_width, img_height)


def generate_whole_tif(pics_dirs, save_path=None, compression='jpeg'):
    if save_path is None:
        save_path = os.path.join(os.path.split(pics_dirs[0])[0], "merge.ome.tif")

    # 遍历目录，根据海德星格式查询shape，文件名等参数, 图像迭代器
    (tiles_down, tiles_across), (img_width, img_height) = HDX_image_prepare(pics_dirs[0])
    if len(pics_dirs) < 1:
        raise Exception(f"input pics_dis < 1. pics_dir:{pics_dirs}")
    t1 = time.time()
    for channel, pics_dir in enumerate(pics_dirs):
        if channel == 0:
            whole_img = gen_HDX_tile_images(pics_dir, tiles_across, tiles_down)
        else:
            tmp_img = gen_HDX_tile_images(pics_dir, tiles_across, tiles_down)
            whole_img = whole_img.bandjoin(tmp_img)
    if len(pics_dirs) == 2:
        whole_img = whole_img.bandjoin_const(0)
    t2 = time.time()
    print(f"{pics_dir} reading all imgs cost {t2 - t1} s")
    # whole_img.tiffsave(save_path, compression=compression, tile=True,
    #                     tile_width=512, tile_height=512, Q=90,
    #                     pyramid=True)
    save_pyramid_tif(whole_img, save_path, compression=compression)
    t3 = time.time()
    print(f"Saving pyramid cost {t3-t2} s")

    return save_path


if __name__ == '__main__':
    print(sys.argv)
    # sys.argv = ['.\\ChipDecodeScript.py', '-d', 'E:\\biomarker_data\\chip4_project', '--mrxs_dir', 'E:\\biomarker_data\\chip4']
    parser = argparse.ArgumentParser(description="merge HDX img_dirs")
    parser.add_argument('--save_path', type=str, help="imgs save dir", default=None)
    parser.add_argument('--img_dir', nargs="+", type=str, help="all img dir path")
    args = parser.parse_args()

    print(args)

    save_path = args.save_path
    img_dir = args.img_dir
    print(img_dir)
    # if save_path is None:
    #     raise Exception("need save path")

    save_path = generate_whole_tif(img_dir, save_path=save_path)
    slide = openslide.open_slide(save_path)
    a = pyvips.Image.new_from_file(save_path)
    print(slide.level_dimensions)
    print(f"{img_dir} has done, save path {save_path}")
