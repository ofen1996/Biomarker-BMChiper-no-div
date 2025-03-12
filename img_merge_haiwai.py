import argparse
import sys
import numpy as np
# from PIL import Image
import os
import tifffile
# import tiffslide
import cv2
# from need.ofen_tool import show_img

## 海外的荧光图像预处理，包含底板，16位->8位，反相+均衡化，然后融合
if __name__ == '__main__':
    print(sys.argv)
    # sys.argv = ['.\\ChipDecodeScript.py', '-d', 'E:\\biomarker_data\\chip4_project', '--mrxs_dir', 'E:\\biomarker_data\\chip4']
    parser = argparse.ArgumentParser(description="merge haiwai fl images")
    parser.add_argument('--save_dir', type=str, help="imgs save dir", default=None)
    parser.add_argument('--board_img', type=str, help="all img dir path")
    parser.add_argument('--fl_img', type=str, help="all img dir path")
    args = parser.parse_args()

    print(args)

    save_dir = args.save_dir
    board_img = args.board_img
    fl_img = args.fl_img

    if save_dir is None:
        save_dir = os.path.split(fl_img)[0]  # 默认为传入第一个路径的上一层

    board_img = tifffile.imread(board_img)
    board_img = board_img.astype(np.float32)
    board_img = board_img/board_img.max()
    board_img = board_img*255
    board_img = board_img.astype(np.uint8)
    board_img = cv2.equalizeHist(cv2.bitwise_not(board_img))

    merge_img = np.zeros((*board_img.shape, 3), dtype=np.uint8)
    merge_img[..., 0] = board_img

    fl_img = tifffile.imread(fl_img)
    fl_img = fl_img/fl_img.max()
    fl_img = fl_img*255
    fl_img = fl_img.astype(np.uint8)

    merge_img[..., 1] = fl_img

    save_path = os.path.join(save_dir, "merge.tif")
    tifffile.imwrite(save_path, merge_img, compression="lzw")
    print(f"merge has done, save path {save_path}")
