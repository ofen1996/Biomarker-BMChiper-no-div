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
os.environ['PATH'] = r"./src/vips-dev-8.15/bin" + ';' + vipsbin + ';' + os.environ['PATH']
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
                pyramid=True, subifd=True)


def generate_matrix(M, N):
    rows, cols = np.meshgrid(np.arange(1, M+1), np.arange(1, N+1), indexing='ij')
    matrix = np.stack((rows, cols), axis=-1)
    return matrix


def gen_HDX_tile_images(pics_dir, right_index):
    r, c = right_index

    pic_name = "IMG{}x{}.tif".format(str(r+1).zfill(3), str(c+1).zfill(3))
    print(pic_name)
    imgs = []
    for p_d in pics_dir:
        img = np.asarray(Image.open(os.path.join(p_d, pic_name)))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img = img[..., 2]
        imgs.append(img)
    if len(imgs) == 2:
        imgs.append(np.zeros_like(img))
    imgs = cv2.merge(imgs)
    # rgb_image = rgb_image.copy(interpretation="srgb")
    return pyvips.Image.new_from_array(imgs)


def HDX_image_prepare(pics_dir):
    img_names = os.listdir(pics_dir[0])

    rows = [int(name[3:6]) for name in img_names if name.endswith(".tif")]
    cols = [int(name[7:10]) for name in img_names if name.endswith(".tif")]
    # 设置矩阵形状
    row, col = max(rows), max(cols)
    shape_r_c = (row, col)

    tmp_im = pyvips.Image.new_from_file(os.path.join(pics_dir[0], img_names[0]))
    img_width, img_height = tmp_im.width, tmp_im.height
    return shape_r_c, (img_width, img_height)


def generate_whole_tif(pics_dirs, save_path=None, compression='jpeg'):
    if save_path is None:
        save_path = os.path.join(os.path.split(pics_dirs[0])[0], os.path.split(pics_dirs[0])[1] + ".ome.tif")

    # 遍历目录，根据海德星格式查询shape，文件名等参数, 图像迭代器
    (row, col), (img_width, img_height) = HDX_image_prepare(pics_dirs)

    t1 = time.time()
    whole_img = pyvips.Image.black(col * img_width, row * img_height, bands=3)
    for r in range(row):
        for c in range(col):
            tile_img = gen_HDX_tile_images(pics_dirs, (r, c))
            whole_img = whole_img.draw_image(tile_img, c * img_width, r * img_height)
    t2 = time.time()
    print(f"reading all imgs cost {t2-t1} s")

    whole_img = whole_img.rot90()  # 海德星图像需要转90度
    save_pyramid_tif(whole_img, save_path, compression=compression)
    t3 = time.time()
    print(f"Saving pyramid cost {t3-t2} s")

    return save_path


if __name__ == '__main__':
    print(sys.argv)
    # sys.argv = ['.\\ChipDecodeScript.py', '-d', 'E:\\biomarker_data\\chip4_project', '--mrxs_dir', 'E:\\biomarker_data\\chip4']
    parser = argparse.ArgumentParser(description="merge HDX img_dirs")
    parser.add_argument('--save_dir', type=str, help="imgs save dir", default=None)
    parser.add_argument('--img_dir', nargs="+", type=str, help="all img dir path")
    args = parser.parse_args()

    print(args)

    save_dir = args.save_dir
    img_dir = args.img_dir
    print(img_dir)
    if len(img_dir)>3:
        raise Exception("路径不能大于3, imgs must < 3")

    if save_dir is None:
        save_dir = os.path.split(img_dir[0])[0]  # 默认为传入第一个路径的上一层

    save_path = generate_whole_tif(img_dir)

    print(f"{img_dir} has done, save path {save_path}")
