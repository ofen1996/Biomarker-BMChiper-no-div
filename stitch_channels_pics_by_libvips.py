import argparse
import sys
import time

# import cv2
import numpy as np
# from PIL import Image
import os
# import tifffile
# import tiffslide
# import cv2
# from need.ofen_tool import show_img
import platform
if platform.system() == 'Windows':
    vipsbin = r'D:\work\python\vips-dev-8.15\bin'
    os.environ['PATH'] = vipsbin + ';' + os.environ['PATH']
import pyvips
# pyvips.leak_set(True)
# import openslide
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
                pyramid=True, properties=True)


def generate_matrix(M, N):
    rows, cols = np.meshgrid(np.arange(1, M+1), np.arange(1, N+1), indexing='ij')
    matrix = np.stack((rows, cols), axis=-1)
    return matrix


def find_HDX_tile_images(pics_dirs, tiles_across, tiles_down):
    tiles = []
    for pics_dir in pics_dirs:
        tiles.append([[pyvips.Image.new_from_file(
            os.path.join(pics_dir, f"IMG{str(y).zfill(3)}x{str(x).zfill(3)}.tif")) for x in range(1, tiles_across + 1)]
            for y in range(1, tiles_down + 1)])

    # 合并bands
    if len(tiles) > 1:
        for y in range(0, tiles_down):
            for x in range(0, tiles_across):
                for band in range(1, len(tiles)):
                    tiles[0][y][x] = tiles[0][y][x].bandjoin(tiles[band][y][x])

    return tiles[0]  # 返回合并后的tiles，tiles[y][x]:提取第y行第x列的图


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


def merge_img(whole_img, sub_img, direct, xref, yref, xsec, ysec, harea=int, bandno=int,
              hwindow=3, overlap_x=int, overlap_y=int):
    try:
        if direct == "vertical":
            whole_img, shift_x_y = whole_img.mosaic(sub_img, direct, xref+overlap_x//2, yref, xsec+overlap_x//2, ysec,
                                                    bandno=bandno,
                                                    harea=harea, dx0=True, dy0=True, hwindow=hwindow)
        else:
            whole_img, shift_x_y = whole_img.mosaic(sub_img, direct, xref, yref+overlap_y//2, xsec, ysec+overlap_y//2,
                                                    bandno=bandno,
                                                    harea=harea, dx0=True, dy0=True, hwindow=hwindow)
        shift_x_y = (shift_x_y['dx0'], shift_x_y['dy0'])
    except Exception as e:
        # 如果没找到足够的信号，则直接根据预定位置缝合
        print(e)
        print(f"img:({yref}, {xref}) not find enough point")
        whole_img = whole_img.merge(sub_img, direct, -xref, -yref)
        shift_x_y = -xref, -yref
    return whole_img, shift_x_y


def analyze_overlap(all_img_tiles, harea=int, bandno=int,
                    hwindow=3, overlap_x=int, overlap_y=int):
    (tiles_down, tiles_across), (img_width, img_height) = (len(all_img_tiles), len(all_img_tiles[0])),\
                                                          (all_img_tiles[0][0].width,  all_img_tiles[0][0].height)

    horizontal_overlap = []
    vertical_overlap = []
    # 抽样检测overlap，抽取中间和4个角共5张图，检测这五张图和周围的overlap
    # sample_locs = [[0.5, 0.5], [0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]
    sample_locs = [[0.3, 0.5], [0.6, 0.4], [0.7, 0.8]]
    for sample_loc in sample_locs:
        loc_y, loc_x = int(sample_loc[0] * tiles_down), int(sample_loc[1] * tiles_across)
        # 下面对选定位置图像的左右，上下4个图分割做overlap估计
        # 上
        try:
            first_y, first_x = loc_y-1, loc_x
            whole_img, shift_x_y = all_img_tiles[first_y][first_x].mosaic(all_img_tiles[first_y+1][first_x], "vertical",
                                                                          0, img_height - overlap_y // 2, 0, overlap_y // 2,
                                                                          bandno=bandno, harea=harea, dx0=True, dy0=True, hwindow=hwindow)
            vertical_overlap.append(shift_x_y)
        except Exception as e:
            # 如果没找到足够的信号，则不采用
            print(e)
            print(f"img:({first_y}, {first_x}) not find enough point")
        # 下
        try:
            first_y, first_x = loc_y, loc_x
            whole_img, shift_x_y = all_img_tiles[first_y][first_x].mosaic(all_img_tiles[first_y+1][first_x], "vertical",
                                                                          0, img_height - overlap_y // 2, 0, overlap_y // 2,
                                                                          bandno=bandno, harea=harea, dx0=True, dy0=True, hwindow=hwindow)
            vertical_overlap.append(shift_x_y)
        except Exception as e:
            # 如果没找到足够的信号，则不采用
            print(e)
            print(f"img:({first_y}, {first_x}) not find enough point")

        # 左
        try:
            first_y, first_x = loc_y, loc_x-1
            whole_img, shift_x_y = all_img_tiles[first_y][first_x].mosaic(all_img_tiles[first_y][first_x+1], "horizontal",
                                                                          img_width - overlap_x // 2, 0, overlap_x // 2, 0,
                                                                          bandno=bandno, harea=harea, dx0=True, dy0=True, hwindow=hwindow)
            horizontal_overlap.append(shift_x_y)
        except Exception as e:
            # 如果没找到足够的信号，则不采用
            print(e)
            print(f"img:({first_y}, {first_x}) not find enough point")
        # 右
        try:
            first_y, first_x = loc_y, loc_x
            whole_img, shift_x_y = all_img_tiles[first_y][first_x].mosaic(all_img_tiles[first_y][first_x+1], "horizontal",
                                                                          img_width - overlap_x // 2, 0, overlap_x // 2, 0,
                                                                          bandno=bandno, harea=harea, dx0=True, dy0=True, hwindow=hwindow)
            horizontal_overlap.append(shift_x_y)
        except Exception as e:
            # 如果没找到足够的信号，则不采用
            print(e)
            print(f"img:({first_y}, {first_x}) not find enough point")

    # 偏移量，分别是垂直方向相邻偏移量和水平方向的偏移量，其中左偏和上偏为正，右偏和下偏为负
    # return vertical_overlap, horizontal_overlap

    # 下面对偏移量求平均
    vertical_overlap_array = np.array([(-shift["dx0"], -shift["dy0"]) for shift in vertical_overlap])
    horizontal_overlap_array = np.array([(-shift["dx0"], -shift["dy0"]) for shift in horizontal_overlap])

    vertical_overlap_mean = np.median(vertical_overlap_array, axis=0)
    horizontal_overlap_mean = np.median(horizontal_overlap_array, axis=0)

    print(f"V-array:{vertical_overlap_array}\n"
          f"H-array:{horizontal_overlap_array}\n"
          f"V-mean:{vertical_overlap_mean}\n"
          f"H-mean:{horizontal_overlap_mean}")

    # 赋值完整的坐标矩阵
    tiles_loc = np.zeros((tiles_down, tiles_across, 2))
    for y in range(tiles_down):
        for x in range(tiles_across):
            # 先纵向偏移，再横向便宜
            tiles_loc[y][x] = y * vertical_overlap_mean + x * horizontal_overlap_mean
            pass
    tiles_loc = np.round(tiles_loc).astype(int)

    # 去除负数
    tiles_loc[..., 0] = tiles_loc[..., 0] - tiles_loc[..., 0].min()
    tiles_loc[..., 1] = tiles_loc[..., 1] - tiles_loc[..., 1].min()
    return tiles_loc


def stitch_whole_tif(pics_dirs, save_path=None,
                     compression='jpeg', stitch_band=0,
                     overlap_x=29, overlap_y=25, harea=15):
    if save_path is None:
        pics_dirs_info = os.path.split(pics_dirs[0])
        save_path = os.path.join(pics_dirs_info[0], f"{pics_dirs_info[1]}.ome.tif")

    # 遍历目录，根据海德星格式查询shape，文件名等参数, 图像迭代器
    (tiles_down, tiles_across), (img_width, img_height) = HDX_image_prepare(pics_dirs[0])
    if len(pics_dirs) < 1:
        raise Exception(f"input pics_dis < 1. pics_dir:{pics_dirs}")
    t1 = time.time()
    # 先提取合并band的图像
    tiles = find_HDX_tile_images(pics_dirs, tiles_across, tiles_down)
    # 缝合
    t1 = time.time()
    # 根据overlap先创建坐标矩阵
    tiles_loc_x = np.arange(0, (img_width-overlap_x)*tiles_across, (img_width-overlap_x))  # 新建一个坐标系
    tiles_loc_y = np.arange(0, (img_height-overlap_y)*tiles_down, (img_height-overlap_y))  # 新建一个坐标系
    loc_ori = np.dstack(np.meshgrid(tiles_loc_x, tiles_loc_y))
    loc = loc_ori.copy()
    # 开始缝合
    # whole_img = pyvips.Image.black(500, 500, bands=2)
    # whole_img = tiles[0][0]

    # show_img(whole_img[1].numpy())
    # start_x = 0

    tiles_loc = analyze_overlap(tiles, harea=harea, bandno=1, hwindow=3, overlap_x=overlap_x, overlap_y=overlap_y)

    whole_img = pyvips.Image.black(tiles_loc[...,0].max()+img_width, tiles_loc[...,1].max()+img_height, bands=3)

    if tiles_loc.shape == 0:
        print("参数错误！")
        return None
    print("Analyze overlap over, start merge imgs")
    for y in range(tiles_down):
        for x in range(tiles_across):
            # direct = "horizontal"
            # print(*-tiles_loc[y, x])
            # whole_img = whole_img.merge(tiles[y][x], direct, *-tiles_loc[y, x])
            whole_img = whole_img.insert(tiles[y][x], *tiles_loc[y, x])
            # show_img(whole_img[1].numpy())

    t2 = time.time()
    print(f"{pics_dirs} reading all imgs cost {t2 - t1} s")
    if len(pics_dirs) == 2:
        whole_img = whole_img.bandjoin_const(0)

    # show_img(whole_img[0].numpy())
    # 写入一些参数，方便后面读取时候能用上
    whole_img = whole_img.copy()
    whole_img.set_type(pyvips.GValue.gint_type, "tiles_n_down", tiles_across)  # 海德星旋转90°，所以长宽都要反转一下
    whole_img.set_type(pyvips.GValue.gint_type, "tiles_n_across", tiles_down)
    whole_img.set_type(pyvips.GValue.gint_type, "tiles_width", img_height)
    whole_img.set_type(pyvips.GValue.gint_type, "tiles_height", img_width)

    # whole_img.tiffsave(save_path, compression=compression, tile=True,
    #                    tile_width=512, tile_height=512, Q=90,
    #                    pyramid=True)
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

    # save_path = stitch_whole_tif(img_dir, save_path=save_path, overlap_x=46, overlap_y=40, compression='lzw')
    save_path = stitch_whole_tif(img_dir, save_path=save_path, overlap_x=31, overlap_y=24, compression='lzw', harea=7)
    # slide = openslide.open_slide(save_path)
    # a = pyvips.Image.new_from_file(save_path)
    # print(slide.level_dimensions)
    print(f"{img_dir} has done, save path {save_path}")
