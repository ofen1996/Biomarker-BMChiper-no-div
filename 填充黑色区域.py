import os.path

import cv2
import tifffile
import numpy as np

image_path = r"E:\test\tmp\scan-1.2025-02-08-07-17-46_chip.tif"
image = tifffile.imread(image_path)
# image = image[:5000,:5000,...]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((3, 3), np.uint8)  # 定义3x3卷积核
mask = cv2.dilate(mask, kernel, iterations=2)  
image[mask==255] = [0,0,0]
result = cv2.inpaint(image, mask, inpaintRadius=13, flags=cv2.INPAINT_TELEA)

save_dir = os.path.split(image_path)[0]
tifffile.imwrite(os.path.join(save_dir, "./result.tif"), result, compression="lzw")