from need.ofen_tool import *
# import os
# os.chdir("../")
import openslide
import cv2
import numpy as np
import tifffile

# test = cv2.imread(r"E:\biomarker_data\no_div_HE\cycle_1\ori_14_9.tif")
# cell_div = cv2.imread("./cell_civ.png")[:, :, 0]
# cell = cv2.imread("./cell.png")
# contours, hierarchy = cv2.findContours(cell_div, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# filter_conts = []
# # 即面积过大滤掉
# area_cutoff = 1.0 * cell_div.shape[0] * cell_div.shape[1] / 15 / 15
# # img_show('1', cv2.drawContours(np.zeros(binary_img.shape), contours, -1, 1, 10))
# for i in range(len(contours)):
#     area = cv2.contourArea(contours[i])
#     if area >= area_cutoff or area <= 30:
#         continue
#     if hierarchy[0][i][3] != -1:
#         continue
#     filter_conts.append(contours[i])
# img = cv2.drawContours(cell, filter_conts, -1, (0, 255, 0), 2)
# show_img(img)


def cut_fov_img(mrxs_path, save_path, camera_resolution=(2448, 2048)):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    slide = openslide.OpenSlide(mrxs_path)

    # 非零起始点位置
    BOUND_X, BOUND_Y = map(int, [slide.properties['openslide.bounds-x'], slide.properties['openslide.bounds-y']])
    # 视场的长宽像素值
    BOUND_WIDTH, BOUND_HEIGHT = map(
        int, [slide.properties['openslide.bounds-width'], slide.properties['openslide.bounds-height']])
    # 视场数目
    FOV_COUNT = int(slide.properties['mirax.NONHIERLAYER_0_SECTION.SCANNED_FOV_COUNT'])
    # 相机固定分辨率
    CAMERA_RESOLUTION = np.asarray(camera_resolution)
    # CAMERA_RESOLUTION = np.asarray((2048, 2048))
    # 视场数目，长，宽。
    FOV_SHAPE = ((BOUND_WIDTH, BOUND_HEIGHT) // (CAMERA_RESOLUTION + 0.00000001) + 1).astype(int)
    # 验证计算的视场长宽与视场数目是否对应，如果不对应，说明计算有误，抛错停止
    if FOV_COUNT % (FOV_SHAPE[0] * FOV_SHAPE[1]) != 0:
        print("FOV_SHAPE: {}".format(FOV_SHAPE))
        print("FOV_COUNT: {}".format(FOV_COUNT))
        raise Exception("FOV_SHAPE can not match the FOV_COUNT! Please Check it!"
                        " 'FOV_SHAPE': {} ; 'FOV_COUNT': {}".format(FOV_SHAPE, FOV_COUNT))
    # 计算得到每一个视场的像素值尺寸
    FOV_PIXES = ((BOUND_WIDTH, BOUND_HEIGHT) / FOV_SHAPE).astype(int)

    if len(os.listdir(save_path)) >= FOV_SHAPE[0] * FOV_SHAPE[1]:
        print(save_path, "\n", "it has cuted, skip it.")
        return FOV_SHAPE[::-1]

    for hi in range(FOV_SHAPE[1]):
        for wi in range(FOV_SHAPE[0]):
            select_part = np.asarray((wi, hi))
            t1 = time.time()
            im = np.asarray(slide.read_region((BOUND_X, BOUND_Y) + FOV_PIXES * select_part, 0, FOV_PIXES))
            t2 = time.time()
            # show_img(im)
            save_name = "ori_{}_{}.tif".format(hi, wi)
            cv2.imwrite(os.path.join(save_path, save_name), im)
            t3 = time.time()
    return FOV_SHAPE[::-1]


def mosaic_pic_line(img, img_part, base_shift=0):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_part_gray = cv2.cvtColor(img_part, cv2.COLOR_BGR2GRAY)

    if base_shift > 0:
        # 匹配模板部分在合成图像右下方
        template_part = img_gray[base_shift: base_shift + img_gray.shape[0], -50:]
    else:
        # 匹配模板部分在合成图像右上方
        template_part = img_gray[:img_gray.shape[0], -50:]
    match_part = img_part_gray[10: -10, :20]
    result = cv2.matchTemplate(template_part, match_part, cv2.TM_SQDIFF_NORMED)
    min_max_loc = cv2.minMaxLoc(result)
    print(min_max_loc)
    new_shift_x_y = np.asarray(min_max_loc[-2][::-1]) - (10, 50)
    # show_img(cv2.hconcat([template_part, match_part]))
    print("new_shift_x_y:{}".format(new_shift_x_y))

    pic_hight = img.shape[0] + abs(new_shift_x_y[0])
    pic_wight = img.shape[1] + img_part.shape[1] - abs(new_shift_x_y[1])
    pic_mosaic = np.zeros((pic_hight, pic_wight, 3), dtype=np.uint8)

    if new_shift_x_y[0] + base_shift > 0:
        # 新图向下偏移
        img_start_loc = [0, 0]
        img_part_start_loc = [new_shift_x_y[0] + base_shift, img.shape[1]]
    else:
        # 新图向上偏移
        img_start_loc = [new_shift_x_y[0] + base_shift, 0]
        img_part_start_loc = [0, img.shape[1]]

    img_part_shift = img_part[:, abs(new_shift_x_y[1]):, :]

    pic_mosaic[img_start_loc[0]: img.shape[0], img_start_loc[1]: img.shape[1]] = img
    pic_mosaic[img_part_start_loc[0]: img_part_start_loc[0] + img_part_shift.shape[0], img_part_start_loc[1]:, :] = img_part_shift

    # M = np.asarray([[1, 0, new_shift_x_y[0]], [0, 1, new_shift_x_y[1]]], dtype=np.float64)
    # shift_part = cv2.warpAffine(img_part, M,
    #                             (img_part_gray.shape[1] + new_shift_x_y[0], img_part_gray.shape[0] + new_shift_x_y[1]))
    # pic_mosaic = cv2.hconcat([img, shift_part])
    return pic_mosaic, new_shift_x_y[0] + base_shift



def match_pic_row(img_before, img):
    # img_before_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_before_gray = img_before
    img_gray = img

    template_part = img_before_gray[:img_before_gray.shape[0], -50:]
    match_part = img_gray[10: -10, :20]
    result = cv2.matchTemplate(template_part, match_part, cv2.TM_SQDIFF_NORMED)
    min_max_loc = cv2.minMaxLoc(result)
    # print(min_max_loc)
    new_shift_x_y = np.asarray(min_max_loc[-2][::-1]) - (10, 50)
    if min_max_loc[0] < 1e-04:  # 低纹理信息，不做识别，取初始位置
        new_shift_x_y = [0, -25]
    # print("new_shift_x_y:{}".format(new_shift_x_y), "value:{}".format(min_max_loc[0]))

    return [new_shift_x_y[0], img_before_gray.shape[1] + new_shift_x_y[1]], min_max_loc[0]


def match_pic_column(img_before, img):
    # img_before = cv2.imread(r"E:\biomarker_data\no_div_HE\cycle_1\ori_6_10.tif")
    # img = cv2.imread(r"E:\biomarker_data\no_div_HE\cycle_1\ori_7_10.tif")
    # img_before_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_before_gray = img_before
    img_gray = img
    #
    # tmp = cv2.vconcat([img_before_gray, img_gray])
    # show_img(tmp)

    template_part = img_before_gray[-50:, :img_before_gray.shape[1]]
    match_part = img_gray[:20, 10: -10]
    result = cv2.matchTemplate(template_part, match_part, cv2.TM_SQDIFF_NORMED)
    min_max_loc = cv2.minMaxLoc(result)
    print(min_max_loc)
    new_shift_x_y = np.asarray(min_max_loc[-2][::-1], dtype=int) - (50, 10)
    print("new_shift_x_y:{}".format(new_shift_x_y), "value:{}".format(min_max_loc[0]))
    if min_max_loc[0] < 1e-04:  # 低纹理信息，不做识别，取初始位置
        new_shift_x_y = [-34, 0]

    return [img_before_gray.shape[0] + new_shift_x_y[0], new_shift_x_y[1]], min_max_loc[0]


class StitchImg:
    def __init__(self, mrxs_path, ori_corners=None, camera_resolution=(2448, 2048)):
        # mrxs_path = r"E:\biomarker_data\no_div_HE\20220927-BG27BN04F6-A4-YQ-XZ-GE-40X-bfh-20220927-134559279.mrxs"
        # pic_dir = r"E:\biomarker_data\no_div_HE\20220927-BG27BN04F6-A4-YQ-XZ-GE-40X-bfh-20220927-134559279_cycle"
        self.camera_resolution = camera_resolution
        self.mrxs_path = mrxs_path
        self.pic_dir = mrxs_path.replace(".mrxs", "-Cut")
        self.img_path = mrxs_path.replace(".mrxs", "-Img")
        # 无缝合图像中4个芯片角点坐标
        self.ori_corners = ori_corners
        self.corners = []

        # 初始化缝合起止
        self.start_pic_index = [0, 0]
        self.end_pic_index = [-1, -1]
        # end_pic_index = [fov_shape[0] - 1, fov_shape[1] - 1]
        # end_pic_index = [28, 22]

    def cut_and_stitch(self):
        print("Start cut pic")
        fov_shape = cut_fov_img(self.mrxs_path, self.pic_dir, self.camera_resolution)
        self.end_pic_index = [fov_shape[0] - 1, fov_shape[1] - 1]
        tmp = cv2.imread(os.path.join(self.pic_dir, "ori_{}_{}.tif".format(self.start_pic_index[0],
                                                                           self.start_pic_index[1])))

        # 计算新的4个角点坐标在哪几个图片内，以及它们在各自图内的位置
        for corner in self.ori_corners:
            pic_index_x_y = [corner[0] // tmp.shape[1], corner[1] // tmp.shape[0]]
            corner_pic_name = "ori_{}_{}.tif".format(pic_index_x_y[1], pic_index_x_y[0])
            rel_loc_x_y = [corner[0] % tmp.shape[1], corner[1] % tmp.shape[0]]

            self.corners.append([corner_pic_name, pic_index_x_y, rel_loc_x_y])

        # 应该选择最小的数字
        self.start_pic_index = [min([x[1][1] for x in self.corners]), min([x[1][0] for x in self.corners])]

        print("First stitch pic is {}, {}".format(self.corners[0][0], self.start_pic_index))

        print("fov size:{}".format(tmp.shape))
        whole_size = (tmp.shape[0] * (self.end_pic_index[0] + 1 - self.start_pic_index[0]),
                      tmp.shape[1] * (self.end_pic_index[1] + 1 - self.start_pic_index[1]))
        print("whole size:{}".format(whole_size))
        whole_img = np.zeros((whole_size[0] + 1000, whole_size[1] + 1000, 3), dtype=np.uint8)
        first_loc = [500, 500]
        all_loc = {}

        for h_i in range(self.start_pic_index[0], self.end_pic_index[0] + 1):
            for w_i in range(self.start_pic_index[1], self.end_pic_index[1] + 1):
                pic_name = "ori_{}_{}.tif".format(h_i, w_i)
                pic_path = os.path.join(self.pic_dir, pic_name)
                img = cv2.imread(pic_path)

                # 判断当前图像的位置：1：起始位置（第一行第一张）
                #                 2：每一列第一张
                #                 3: 第一行图像
                #                 4：其余位置
                if h_i == self.start_pic_index[0] and w_i == self.start_pic_index[1]:
                    start_loc = first_loc

                elif w_i == self.start_pic_index[1]:
                    # 找上一张图像做匹配参考
                    pic_name_before = "ori_{}_{}.tif".format(h_i - 1, w_i)
                    pic_path_before = os.path.join(self.pic_dir, pic_name_before)
                    img_before = cv2.imread(pic_path_before)
                    # 计算与上一张图的位置偏移
                    shift_h_w, _ = match_pic_column(img_before, img)
                    start_loc_before = all_loc[pic_name_before]
                    start_loc = [start_loc_before[0] + shift_h_w[0], start_loc_before[1] + shift_h_w[1]]
                    pass
                elif h_i == self.start_pic_index[0]:
                    # 找前一张图像做匹配参考
                    pic_name_before = "ori_{}_{}.tif".format(h_i, w_i - 1)
                    pic_path_before = os.path.join(self.pic_dir, pic_name_before)
                    img_before = cv2.imread(pic_path_before)
                    # 计算与前一张图的位置偏移
                    shift_h_w, _ = match_pic_row(img_before, img)
                    start_loc_before = all_loc[pic_name_before]
                    start_loc = [start_loc_before[0] + shift_h_w[0], start_loc_before[1] + shift_h_w[1]]
                    pass
                else:
                    # 找上一张图像做匹配参考
                    pic_name_before = "ori_{}_{}.tif".format(h_i - 1, w_i)
                    pic_path_before = os.path.join(self.pic_dir, pic_name_before)
                    img_before = cv2.imread(pic_path_before)
                    # 计算与上一张图的位置偏移
                    shift_h_w, match_point_col = match_pic_column(img_before, img)
                    start_loc_before = all_loc[pic_name_before]
                    start_loc_1 = [start_loc_before[0] + shift_h_w[0], start_loc_before[1] + shift_h_w[1]]

                    # 找前一张图像做匹配参考
                    pic_name_before = "ori_{}_{}.tif".format(h_i, w_i - 1)
                    pic_path_before = os.path.join(self.pic_dir, pic_name_before)
                    img_before = cv2.imread(pic_path_before)
                    # 计算与前一张图的位置偏移
                    shift_h_w, match_point_row = match_pic_row(img_before, img)
                    start_loc_before = all_loc[pic_name_before]
                    start_loc_2 = [start_loc_before[0] + shift_h_w[0], start_loc_before[1] + shift_h_w[1]]

                    start_loc = [(start_loc_1[0] + start_loc_2[0]) // 2, (start_loc_1[1] + start_loc_2[1]) // 2]
                    pass

                whole_img[start_loc[0]: start_loc[0] + img.shape[0], start_loc[1]: start_loc[1] + img.shape[1]] = img
                all_loc[pic_name] = start_loc
                print(pic_name, "finished, start loc:{}".format(start_loc))

        # # 画边界线，测试使用
        # for start_loc in all_loc.values():
        #     cv2.rectangle(whole_img, start_loc[::-1], (start_loc[1] + img.shape[1], start_loc[0] + img.shape[0]), (0, 255, 0), 4)

        # show_img(whole_img)

        print("start save")
        if not os.path.exists(self.img_path):
            os.mkdir(self.img_path)
        # cv2.imwrite("./test.tif", whole_img)
        # tifffile.imwrite("./test.tif", whole_img, bigtiff=True)
        # np.save('./test_compress.npy', whole_img)
        # tifffile.imwrite("./test_compress.tif", whole_img, compression='jpeg')
        # save_json("./all_loc.json", all_loc)
        try:
            tifffile.imwrite(os.path.join(self.img_path, 'compress_40x.tif'), whole_img, compression="jpeg")
            # with tifffile.TiffWriter(os.path.join(self.img_path, 'compress_40x.tif'),
            #                          bigtiff=True, ome=True) as tif:
            #     # tif.write(data=whole_img, tile=(256, 256), photometric='rgb', compression='jpeg', dtype=np.uint8)
            #     tif.write(data=whole_img, tile=(256, 256), subifds=3, photometric='rgb', compression='jpeg', dtype=np.uint8)
            #     tif.write(data=whole_img[::2, ::2, :], subfiletype=1, tile=(256, 256), photometric='rgb', compression='jpeg', dtype=np.uint8)
            #     tif.write(data=whole_img[::4, ::4, :], subfiletype=1, tile=(256, 256), photometric='rgb', compression='jpeg', dtype=np.uint8)
            #     tif.write(data=whole_img[::16, ::16, :], subfiletype=1, tile=(256, 256), photometric='rgb', compression='jpeg', dtype=np.uint8)
        except Exception as e:
            print("Save whole image ERROR : {}. ".format(e))
            print("Change to save -compress_40x.npy-")
            # np.savez_compressed(os.path.join(self.img_path, 'compress_40x.tif'), whole_img=whole_img)

        # 返回4个角点的新坐标和level=2的图像

        level_2_path = os.path.join(self.img_path, 'level-2.tif')
        level_2_img = whole_img[::4, ::4, :]
        tifffile.imwrite(level_2_path, level_2_img, compression="jpeg")

        # 找到新的角点，进行映射变换
        new_corners = []
        for corner in self.corners:
            tmp_corner = np.asarray(all_loc[corner[0]]) + corner[2][::-1]
            new_corners.append(tmp_corner.tolist()[::-1])

        level = 2
        tmp_corners = np.asarray(new_corners) // (2**level)

        # 画一下提示点
        pass

        # 计算长宽
        img_w = np.sqrt(np.power(tmp_corners[0][0]-tmp_corners[1][0], 2.0)+np.power(tmp_corners[0][1]-tmp_corners[1][1], 2.0))
        img_h = np.sqrt(np.power(tmp_corners[0][0]-tmp_corners[3][0], 2.0)+np.power(tmp_corners[0][1]-tmp_corners[3][1], 2.0))
        img_w = round(img_w)
        img_h = round(img_h)

        # 变换矩阵
        desc_box = [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]
        M = cv2.getPerspectiveTransform(np.array(tmp_corners, dtype='float32'), np.array(desc_box, dtype='float32'))
        # 透视变换
        level_2_img_chip = cv2.warpPerspective(level_2_img, M, (img_w, img_h))
        print(f"reg_box={tmp_corners}, desc_box={desc_box}")
        tifffile.imwrite(os.path.join(self.img_path, 'level-2-chip.tif'), level_2_img_chip, compression="jpeg")

        for key, value in all_loc.items():
            all_loc[key] = list(map(int, value))
        config_path = os.path.join(self.img_path, '40x-config.json')
        config_data = {"all_loc": all_loc, "new_corners": new_corners}
        save_json(config_path, config_data)

        return new_corners, os.path.join(self.img_path, 'level-2-chip.tif')


def stitch_from_pic(pic_dir, fov_shape, save_dir, start_pic_index=(0, 0)):
    end_pic_index = [fov_shape[0] - 1, fov_shape[1] - 1]
    tmp = cv2.imread(os.path.join(pic_dir, "ori_{}_{}.tif".format(start_pic_index[0],
                                                                  start_pic_index[1])))

    print("fov size:{}".format(tmp.shape))
    whole_size = (tmp.shape[0] * (end_pic_index[0] + 1 - start_pic_index[0]),
                  tmp.shape[1] * (end_pic_index[1] + 1 - start_pic_index[1]))
    print("whole size:{}".format(whole_size))
    whole_img = np.zeros((whole_size[0] + 1000, whole_size[1] + 1000, 3), dtype=np.uint8)
    first_loc = [500, 500]
    all_loc = {}

    for h_i in range(start_pic_index[0], end_pic_index[0] + 1):
        for w_i in range(start_pic_index[1], end_pic_index[1] + 1):
            pic_name = "ori_{}_{}.tif".format(h_i, w_i)
            pic_path = os.path.join(pic_dir, pic_name)
            img = cv2.imread(pic_path)

            # 判断当前图像的位置：1：起始位置（第一行第一张）
            #                 2：每一列第一张
            #                 3: 第一行图像
            #                 4：其余位置
            if h_i == start_pic_index[0] and w_i == start_pic_index[1]:
                start_loc = first_loc

            elif w_i == start_pic_index[1]:
                # 找上一张图像做匹配参考
                pic_name_before = "ori_{}_{}.tif".format(h_i - 1, w_i)
                pic_path_before = os.path.join(pic_dir, pic_name_before)
                img_before = cv2.imread(pic_path_before)
                # 计算与上一张图的位置偏移
                shift_h_w, _ = match_pic_column(img_before, img)
                start_loc_before = all_loc[pic_name_before]
                start_loc = [start_loc_before[0] + shift_h_w[0], start_loc_before[1] + shift_h_w[1]]
                pass
            elif h_i == start_pic_index[0]:
                # 找前一张图像做匹配参考
                pic_name_before = "ori_{}_{}.tif".format(h_i, w_i - 1)
                pic_path_before = os.path.join(pic_dir, pic_name_before)
                img_before = cv2.imread(pic_path_before)
                # 计算与前一张图的位置偏移
                shift_h_w, _ = match_pic_row(img_before, img)
                start_loc_before = all_loc[pic_name_before]
                start_loc = [start_loc_before[0] + shift_h_w[0], start_loc_before[1] + shift_h_w[1]]
                pass
            else:
                # 找上一张图像做匹配参考
                pic_name_before = "ori_{}_{}.tif".format(h_i - 1, w_i)
                pic_path_before = os.path.join(pic_dir, pic_name_before)
                img_before = cv2.imread(pic_path_before)
                # 计算与上一张图的位置偏移
                shift_h_w, match_point_col = match_pic_column(img_before, img)
                start_loc_before = all_loc[pic_name_before]
                start_loc_1 = [start_loc_before[0] + shift_h_w[0], start_loc_before[1] + shift_h_w[1]]

                # 找前一张图像做匹配参考
                pic_name_before = "ori_{}_{}.tif".format(h_i, w_i - 1)
                pic_path_before = os.path.join(pic_dir, pic_name_before)
                img_before = cv2.imread(pic_path_before)
                # 计算与前一张图的位置偏移
                shift_h_w, match_point_row = match_pic_row(img_before, img)
                start_loc_before = all_loc[pic_name_before]
                start_loc_2 = [start_loc_before[0] + shift_h_w[0], start_loc_before[1] + shift_h_w[1]]

                start_loc = [(start_loc_1[0] + start_loc_2[0]) // 2, (start_loc_1[1] + start_loc_2[1]) // 2]
                pass

            whole_img[start_loc[0]: start_loc[0] + img.shape[0], start_loc[1]: start_loc[1] + img.shape[1]] = img
            all_loc[pic_name] = start_loc
            print(pic_name, "finished, start loc:{}".format(start_loc))

    # # 画边界线，测试使用
    # for start_loc in all_loc.values():
    #     cv2.rectangle(whole_img, start_loc[::-1], (start_loc[1] + img.shape[1], start_loc[0] + img.shape[0]), (0, 255, 0), 4)

    # show_img(whole_img)

    print("start save")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # cv2.imwrite("./test.tif", whole_img)
    # tifffile.imwrite("./test.tif", whole_img, bigtiff=True)
    # np.save('./test_compress.npy', whole_img)
    # tifffile.imwrite("./test_compress.tif", whole_img, compression='jpeg')
    # save_json("./all_loc.json", all_loc)
    try:
        tifffile.imwrite(os.path.join(save_dir, 'compress_40x.tif'), whole_img, compression="jpeg")
        # with tifffile.TiffWriter(os.path.join(self.img_path, 'compress_40x.tif'),
        #                          bigtiff=True, ome=True) as tif:
        #     # tif.write(data=whole_img, tile=(256, 256), photometric='rgb', compression='jpeg', dtype=np.uint8)
        #     tif.write(data=whole_img, tile=(256, 256), subifds=3, photometric='rgb', compression='jpeg', dtype=np.uint8)
        #     tif.write(data=whole_img[::2, ::2, :], subfiletype=1, tile=(256, 256), photometric='rgb', compression='jpeg', dtype=np.uint8)
        #     tif.write(data=whole_img[::4, ::4, :], subfiletype=1, tile=(256, 256), photometric='rgb', compression='jpeg', dtype=np.uint8)
        #     tif.write(data=whole_img[::16, ::16, :], subfiletype=1, tile=(256, 256), photometric='rgb', compression='jpeg', dtype=np.uint8)
    except Exception as e:
        print("Save whole image ERROR : {}. ".format(e))
        # np.savez_compressed(os.path.join(self.img_path, 'compress_40x.tif'), whole_img=whole_img)

    # 返回4个角点的新坐标和level=2的图像

    level_2_path = os.path.join(save_dir, 'level-2.tif')
    level_2_img = whole_img[::4, ::4, :]
    tifffile.imwrite(level_2_path, level_2_img, compression="jpeg")


if __name__ == '__main__':
    pass
    stitch_from_pic(r"E:\0920-20220803-BG27BN04F4-A3-1-cut", (12, 12), r"E:\0920-20220803-BG27BN04F4-A3-1-img")
    # cut_fov_img(r"E:\biomarker_data\no_div_HE\20x-test\0920-20220803-BG27BN04F4-A3-1.mrxs",
    #             r"E:\biomarker_data\no_div_HE\20x-test\0920-20220803-BG27BN04F4-A3-1-Cut",
    #             camera_resolution=(2048, 2048))
