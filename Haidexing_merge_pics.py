import argparse
import sys
import numpy as np
from PIL import Image
import os
import tifffile
import tiffslide
import cv2
from need.ofen_tool import show_img

## 海德星的小图拼接成大图


def gen_ome_tif(save_path, data, tile_size_y_x, shape_r_c_pix,
                sub_rate=2, pyramid_levels=4,
                compression='jpeg'):

    with tifffile.Timer():
        with tifffile.TiffWriter(save_path, bigtiff=True) as tif:
            tif.write(
                data=data,
                dtype='uint8',
                shape=(shape_r_c_pix[0], shape_r_c_pix[1], 3),
                tile=tile_size_y_x,
                compression=compression,
                maxworkers=2,
                subifds=pyramid_levels,
            )
            for level in range(pyramid_levels):
                mag = sub_rate ** (level + 1)
                res = 1e4 / mag / 0.1
                tmp_slide = tiffslide.open_slide(save_path)
                sub_image = tmp_slide.read_region((0, 0), level, tmp_slide.level_dimensions[level], as_array=True)
                # tile = (tile_size[0] // mag, tile_size[1] // mag)
                # shape = (tile[0] * (shape_r_c_pix[0] // tile[0] // mag),
                #          tile[1] * (shape_r_c_pix[1] // tile[1] // mag), 3)
                sub_image = sub_image[::sub_rate, ::sub_rate, ...]
                tif.write(
                    data=sub_image,
                    subfiletype=1,  # Set bit 0 if the image is a reduced-resolution version
                    # resolution=(res, res),
                    compression=compression,
                    dtype='uint8',
                    # shape=(res, res),
                    tile=(512, 512),
                )

            # thumbnail = sub_image[::4, ::4, ...]
            # tif.write(thumbnail, metadata={'Name': 'thumbnail'})


def generate_matrix(M, N):
    rows, cols = np.meshgrid(np.arange(1, M+1), np.arange(1, N+1), indexing='ij')
    matrix = np.stack((rows, cols), axis=-1)
    return matrix


def gen_HDX_tile_images(pics_dir, right_index, gray_style=False):
    # if gray_style:
    #     whole_size = (de_pixes[1] * col, de_pixes[0] * row)
    # else:
    #     whole_size = (de_pixes[1] * col, de_pixes[0] * row, 3)
    # whole_img = np.zeros(whole_size, dtype=np.uint8)
    for y_id in range(right_index.shape[0]):
        for x_id in range(right_index.shape[1]):
            r, c = right_index[y_id, x_id]

            pic_name = "IMG{}x{}.tif".format(str(r).zfill(3), str(c).zfill(3))
            # print(pic_name)
            imgs = []
            for p_d in pics_dir:
                img = np.asarray(Image.open(os.path.join(p_d, pic_name)))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                img = img[..., 2]
                imgs.append(img)
            imgs = cv2.merge(imgs)
            imgs = cv2.rotate(imgs, cv2.ROTATE_90_CLOCKWISE)  # 旋转角度

            yield imgs


def HDX_image_prepare(pics_dir):
    img_names = os.listdir(pics_dir[0])

    rows = [int(name[3:6]) for name in img_names if name.endswith(".tif")]
    cols = [int(name[7:10]) for name in img_names if name.endswith(".tif")]
    # 设置矩阵形状
    row, col = max(rows), max(cols)
    shape_r_c = (row, col)
    # 生成矩阵
    result_matrix = generate_matrix(row, col)
    # 顺时针偏转90°
    right_index = np.rot90(result_matrix, -1)  # 存着正确的序号的矩阵

    data = gen_HDX_tile_images(pics_dir, right_index, gray_style=False)
    return shape_r_c, data


def generate_whole_tif(pics_dirs, save_path=None, pixes=(2048, 2448), gray_style=False, compression='jpeg'):
    if save_path is None:
        save_path = os.path.join(os.path.split(pics_dirs[0])[0], os.path.split(pics_dirs[0])[1] + ".ome.tif")

    # 遍历目录，根据海德星格式查询shape，文件名等参数, 图像迭代器
    shape_r_c, data = HDX_image_prepare(pics_dirs)

    sub_rate = 2
    pyramid_levels = 4  # 金字塔等级
    tile_size_y_x = pixes
    shape_r_c_pix = np.asarray(shape_r_c) * tile_size_y_x  # 0级尺寸
    shape_r_c_pix = shape_r_c_pix[::-1]  # 图像需要转90°，所以行列互换
    tile_size_y_x = tile_size_y_x[::-1]

    gen_ome_tif(save_path, data, tile_size_y_x, shape_r_c_pix,
                sub_rate=sub_rate, pyramid_levels=pyramid_levels,
                compression=compression)

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
