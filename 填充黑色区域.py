import cv2
import tifffile
import numpy as np

image = tifffile.imread("./DG04DN02F3-B2.tif")
# image = image[:5000,:5000,...]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((3, 3), np.uint8)  # 定义3x3卷积核
mask = cv2.dilate(mask, kernel, iterations=2)  
image[mask==255] = [0,0,0]
result = cv2.inpaint(image, mask, inpaintRadius=13, flags=cv2.INPAINT_TELEA)
tifffile.imwrite("./test.tif", result, compression="jpeg")