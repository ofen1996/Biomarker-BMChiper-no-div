import copy
import numpy as np
import cv2
import openslide
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = 3000000000



# 定义图片展示函数
ShowImageType = 1
def img_show(name, img):
    if ShowImageType == 0:
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        plt.imshow(img)
        plt.title(name)
        plt.show()


class MRXSBase():
    def __init__(self, slide_file):
        # 读取切片文件
        self.slide = openslide.OpenSlide(slide_file)
        # init slide
        self.init_slide()

        pass

    def init_slide(self):
        # 图像区初始化
        self.slide_x = int(self.slide.properties['openslide.bounds-x'])
        self.slide_y = int(self.slide.properties['openslide.bounds-y'])
        self.slide_width = int(self.slide.properties['openslide.bounds-width'])
        self.slide_height = int(self.slide.properties['openslide.bounds-height'])
        # self.slide_height = 56298

        # level水平初始化
        # level
        self.levels = [0, 1, 2, 3, 4, 5]
        # 缩放倍数
        self.level_zoom = np.zeros(len(self.levels),dtype='float')
        # 对应水平下的图片WIDTH与HEIGHT
        self.level_w_h = np.zeros((len(self.levels),2),dtype='int')
        for i in self.levels:
            # 缩放倍数
            self.level_zoom[i] = np.power(2.0,i)
            # 对应水平下的图片WIDTH与HEIGHT
            self.level_w_h[i,0] = int(self.slide_width/self.level_zoom[i])
            self.level_w_h[i,1] = int(self.slide_height/self.level_zoom[i])
        pass

    # 从指定的level水平上
    # 在(img_box[0],img_box[1])处，截取一个W与H分别为img_box[2]与img_box[3]的区域
    # 注意：1) x,y的起始点是以openslide.bounds-x与openslide.bounds-y为原点的
    #      2) W与H是基于level级别的
    # 返回np_img
    def mrxs_crop(self, img_box, level):
        # 基于slide_x,slide_y，以及level水平，对x,y进行修正
        box_x = self.slide_x + self.level_zoom[level]*img_box[0]
        box_y = self.slide_y + self.level_zoom[level]*img_box[1]
        box_x = int(box_x)
        box_y = int(box_y)
        # crop img
        tile = np.array(self.slide.read_region((box_x, box_y), level, (img_box[2], img_box[3])))
        return tile

    def extract_img_by_level(self, level):
        return self.slide.read_region((self.slide_x, self.slide_y), level=level, size=self.level_w_h[level])




# 对荧光图片的荧光值进行增强处理
def adjust_brightness(img,black=[2,2,2],gamma=[1.69,1.69,1.69],white=[30,30,30]):
    r, g, b, a = cv2.split(img)
    # r = np.where(r<black[0],0,r)
    g = np.where(g<black[1],0,g)
    b = np.where(b<black[2],0,b)
    # r = np.power(r*1.0*white[0],gamma[0])
    g = np.power(g*1.0*white[1],gamma[1])
    b = np.power(b*1.0*white[2],gamma[2])
    r = np.minimum(r, 255)
    g = np.minimum(g, 255)
    b = np.minimum(b, 255)
    r=r.astype(np.uint8)
    g=g.astype(np.uint8)
    b=b.astype(np.uint8)
    print("r.min:",r.max())
    print("g.min:",g.max())
    print("b.min:",b.max())

    new_img = Image.fromarray(np.dstack((b,g,r)))
    return np.array(new_img)
    pass


if __name__ == '__main__':
    # 读取图片信息
    flu_file = 'D:\\delete\\flu\\flu_8889_bc1_40X.mrxs'

    # create
    mb = MRXSBase(slide_file=flu_file)

    # 提取一个图片
    nbox = [0,0,8000,8000]
    np_img = mb.mrxs_crop(nbox,1)
    print(np_img.shape)
    img_show("img", np_img)

    fluimg = adjust_brightness(np_img, [2, 8, 8], [1.0, 1.0, 1.0], [20, 20, 20])
    img_show("fluimg", fluimg)


    pass


