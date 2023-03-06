import numpy as np
import openslide
import tifffile
from need.ofen_tool import *
from need.KpDetectByYolo import MyDetector
from need.config import conf

detector = MyDetector("./model/best.onnx")
std_edge_size = (2248, 2648)


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
    # 计算得到每一个视场的像素值尺寸
    FOV_PIXES = ((BOUND_WIDTH, BOUND_HEIGHT) / FOV_SHAPE).astype(int)

    if len(os.listdir(save_path)) >= FOV_SHAPE[0] * FOV_SHAPE[1]:
        print(save_path, "\n", "it has cuted, skip it.")
        return FOV_SHAPE[::-1], FOV_PIXES[::-1]

    for hi in range(FOV_SHAPE[1]):
        for wi in range(FOV_SHAPE[0]):
            select_part = np.asarray((wi, hi))
            t1 = time.time()
            im = np.asarray(slide.read_region((BOUND_X, BOUND_Y) + FOV_PIXES * select_part, 0, FOV_PIXES))
            t2 = time.time()
            # show_img(im)
            save_name = "ori_{}_{}.tif".format(hi, wi)
            tifffile.imwrite(os.path.join(save_path, save_name), im)
            t3 = time.time()
    return FOV_SHAPE[::-1], FOV_PIXES[::-1]


class StdCircles:
    def __init__(self, ori_size, shape, d, r):
        self.circles_img, self.circles_mask, self.circle_centers = self.std_circles(ori_size, shape, d, r)
        # tmp = np.where(self.circles_mask > 0, 255, 0).astype(np.uint8)
        # edge_rect_dilate_a = int(conf.conf.get("std-template", "edge_rect_dilate_a"))
        # edge_rect_dilate_b = int(conf.conf.get("std-template", "edge_rect_dilate_b"))
        # tmp_a = cv2.dilate(tmp, cv2.getStructuringElement(cv2.MORPH_DILATE, (edge_rect_dilate_a, edge_rect_dilate_a)))
        # tmp_b = cv2.dilate(tmp, cv2.getStructuringElement(cv2.MORPH_RECT, (edge_rect_dilate_b, edge_rect_dilate_b)))
        # self.edge_rect = tmp_b - tmp_a
        cv2.imwrite("test_circles.tif", self.circles_img)
        cv2.imwrite("test_circles_mask.tif", self.circles_mask)
        # cv2.imwrite("edge_rect.tif", self.edge_rect)

        # 计算首个分割点位置
        temp_delta_y = (self.circle_centers[30, 37][1] - self.circle_centers[30, 35][1]) // 2
        temp_delta_x = (self.circle_centers[32, 34][0] - self.circle_centers[30, 35][0]) // 2
        self.std_first_div_point = (self.circle_centers[30, 35] + (temp_delta_x, temp_delta_y))[::-1]
        # self.std_first_div_point = (self.circle_centers[30, 35] + (26, 30))[::-1]

        # 标准单个区域模板
        # 先计算首个分割点的对角点位置
        tmp_corner_div_point = (self.circle_centers[61, 71] + (temp_delta_x, temp_delta_y))[::-1]
        # print(tmp_corner_div_point - self.std_first_div_point)
        self.std_mask_size = (tmp_corner_div_point - self.std_first_div_point).tolist()
        # self.std_mask_size = [1081, 1076]
        # conf.std_distance = self.std_mask_size[0]
        self.std_mask = self.circles_img[self.std_first_div_point[0]:self.std_first_div_point[0] + self.std_mask_size[0],
                                         self.std_first_div_point[1]:self.std_first_div_point[1] + self.std_mask_size[1]]
        self.std_mask_centers = self.circle_centers[32: 32 + 30, 37: 37 + 35] - self.std_first_div_point[::-1]

    @staticmethod
    def std_circles(ori_size, shape, d, r, tile_limit=(46, 46)):
        # std_circles((1200, 1200), (30, 35), 35, 10)
        size = (ori_size[0] + 30, ori_size[1] + 30)
        circles_img = np.zeros(size, dtype=np.uint8)
        circles_mask = np.zeros(size, dtype=np.uint8)
        delta_x = d
        delta_y = d * 0.5 * 3 ** 0.5
        total_circles_shape = (int(size[0] / delta_y) + 1, int(size[1] / delta_x) + 1)[::-1]
        # first_circle_loc = (np.asarray(size) - (np.asarray(shape) - 1) * np.asarray((delta_x, delta_y))) // 2
        # first_circle_loc = np.asarray((delta_x, delta_y), dtype=int)
        first_circle_loc = np.asarray((conf.std_edge_size[1], conf.std_edge_size[0]), dtype=int)

        circle_centers = np.zeros((*total_circles_shape, 2), dtype=int)
        for y in range(total_circles_shape[1]):
            for x in range(total_circles_shape[0]):

                rate = 0.5
                if y % (shape[1] + 1) in (1, shape[1]) or x % (shape[0] + 1) in (1, shape[0]):
                    rate *= 2  # 对边缘最近的一行的权重提高，这样在计算时候可以让匹配尽量贴近边缘，避免边缘不完整图像匹配出错
                first_loc = first_circle_loc
                if y % 2 == 0:
                    first_loc = first_circle_loc - (d / 2, 0)
                tmp_center = (first_loc + (x * delta_x, y * delta_y)).astype(int)
                # circle_centers.append(tmp_center.astype(int))
                circle_centers[x, y, :] = tmp_center
                if y >= (shape[1] + 1) * tile_limit[1] or x >= (shape[0] + 1) * tile_limit[0]:
                    # 超过芯片边界的不要
                    continue
                if y % (shape[1] + 1) == 0 or x % (shape[0] + 1) == 0:
                    # 不同块的分割线
                    continue
                circles_mask = cv2.circle(circles_mask, tuple(tmp_center), r, int(255 * rate), -1)
                circles_img = cv2.circle(circles_img, tuple(tmp_center), r, 255, 1)
        # print(circle_centers)
        # for center in circle_centers.reshape(total_circles_shape[0] * total_circles_shape[1], -1):
        #     circles_mask = cv2.circle(circles_mask, tuple(center), r, 255, -1)
        #     circles_img = cv2.circle(circles_img, tuple(center), r, 255, 1)
        # show_img(circles_img)
        # show_img(circles_mask)
        circles_mask = cv2.bitwise_not(circles_mask)
        return [circles_img[:ori_size[0], :ori_size[1]],
                circles_mask[:ori_size[0], :ori_size[1]],
                circle_centers]

def two_point_dist(pt1, pt2):
    pow1 = np.power(pt2[0] - pt1[0], 2.0)
    pow2 = np.power(pt2[1] - pt1[1], 2.0)
    # print(pt1,pt2,pow1,pow2)
    return np.sqrt(pow1 + pow2)


def M_matrix(scr_rect, dst_rect):
    rect = np.array(scr_rect, dtype='float32')
    dst = np.array(dst_rect, dtype='float32')
    # 变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    # return
    return M


def calculate_M(reg_box):
    reg_box = np.asarray(reg_box)
    mrxs_rect = reg_box - reg_box[0]
    # mrxs_rect = copy.deepcopy(tmp_reg_box)
    img_wh = [0, 0]
    img_wh[0] = round(two_point_dist(mrxs_rect[0], mrxs_rect[1]))
    img_wh[1] = round(two_point_dist(mrxs_rect[0], mrxs_rect[3]))
    nimg_rect = [[0, 0], [img_wh[0] - 1, 0], [img_wh[0] - 1, img_wh[1] - 1], [0, img_wh[1] - 1]]

    # 计算M变换矩阵
    M = M_matrix(mrxs_rect, nimg_rect)
    return M


def gen_index_by_reg_box(reg_box, img_shape):
    reg_box = np.asarray(reg_box)
    x_start = reg_box[0][0] // img_shape[1]
    y_start = reg_box[0][1] // img_shape[0]

    x_end = reg_box[2][0] // img_shape[1]
    y_end = reg_box[2][1] // img_shape[0]

    all_index_y_x = []
    for y in range(y_start, y_end + 1):
        for x in range(x_start, x_end + 1):
            all_index_y_x.append([y, x])
    return [[x_start, y_start], [x_end, y_end]], all_index_y_x


def new_stitch(pics_dir, reg_box, pic_shape=(2048, 2448),save_dir=None):
    stitch_json = {"reg_box": reg_box}
    M = calculate_M(reg_box)
    stitch_json["M"] = M.tolist()
    std_circle = StdCircles(conf.whole_img_size, (30, 35), 17.565, 7)
    print("---end draw std circles")
    # stitch_img = cv2.cvtColor(std_circle.circles_mask, cv2.COLOR_GRAY2BGR)
    stitch_img = np.zeros((*conf.whole_img_size, 3), dtype=np.uint8)

    # whole_start_loc = reg_box[0]
    # whole_end_loc = reg_box[2]
    print("---end draw std stitch_img")
    print("start match img---")

    _, all_index_y_x = gen_index_by_reg_box(reg_box, pic_shape)
    for index_y_x in all_index_y_x:

        index_y, index_x = index_y_x

        print("start match ori_{}_{}.tif".format(index_y, index_x))
        img_ori = cv2.imread(os.path.join(pics_dir, "ori_{}_{}.tif".format(index_y, index_x)))
        img = cv2.warpPerspective(img_ori, M, img_ori.shape[:2][::-1])
        # img = img_ori.copy()
        out_img, centers = detector.detect(img, 0.4)

        if not centers:
            print("Warnning: ori_{}_{}.tif has 0 Key Point, Skip it.".format(index_y, index_x))
            continue

        max_match_kp = centers[0][:2]

        # 计算max_match_kp在整个图像中的空间坐标
        max_match_kp_loc = [index_x * img_ori.shape[1] + max_match_kp[0], index_y * img_ori.shape[0] + max_match_kp[1]]
        # 计算max_match_kp相对芯片左上角的坐标
        max_match_kp_rel = np.asarray(max_match_kp_loc) - reg_box[0]

        # 下面估算预测的关键点在std_mask底板中的位置
        distance = (np.asarray(reg_box[2]) - reg_box[0]) // 46
        tile_index_x, tile_index_y = np.round(max_match_kp_rel / distance).astype(int)
        kp_loc = std_circle.circle_centers[tile_index_x * 31 + 1, tile_index_y * 36 + 1]  # 找到对应块的第一个圆心坐标(x,y)，近似对应kp位置
        # show_img(std_circle.circles_mask[kp_loc[1]:kp_loc[1]+400, kp_loc[0]:kp_loc[0]+400])

        # 从mask里面截取模板template，然后精准匹配
        template_start_loc = np.asarray(kp_loc) - max_match_kp
        template = std_circle.circles_mask[template_start_loc[1]-200:template_start_loc[1]+img_ori.shape[0]+200,
                                           template_start_loc[0]-200:template_start_loc[0]+img_ori.shape[1]+200]

        match_result = cv2.matchTemplate(template, img[..., conf.stitch_channal], cv2.TM_SQDIFF)
        match_shift = cv2.minMaxLoc(match_result)[2] - np.asarray([200, 200])
        real_loc = template_start_loc + match_shift

        stitch_json["ori_{}_{}.tif".format(index_y, index_x)] = real_loc.tolist()
        # 覆写mask
        # 先丢弃部分因仿射变换带来的黑边
        crop_rate = 0.003
        # crop_rate = 0
        img_crop = img[int(img.shape[0] * crop_rate): img.shape[0]-int(img.shape[0] * crop_rate),
                       int(img.shape[1] * crop_rate): img.shape[1]-int(img.shape[1] * crop_rate)]
        real_loc_crop = real_loc + np.asarray([int(img.shape[1] * crop_rate), int(img.shape[0] * crop_rate)])
        std_circle.circles_mask[real_loc_crop[1]:real_loc_crop[1] + img_crop.shape[0],
                                real_loc_crop[0]:real_loc_crop[0] + img_crop.shape[1]] += img_crop[..., conf.stitch_channal]//2
        stitch_img[real_loc_crop[1]:real_loc_crop[1] + img_crop.shape[0],
                   real_loc_crop[0]:real_loc_crop[0] + img_crop.shape[1]] = img_crop

    if save_dir is None:
        save_dir = os.path.split(pics_dir)[0]
    print("start save {}".format(os.path.join(save_dir, r"new_stitch_img_mask.tif")))
    tifffile.imwrite(os.path.join(save_dir, "new_stitch_img_mask.tif"), std_circle.circles_mask, compression="jpeg")
    print("start save {}".format(os.path.join(save_dir, r"new_stitch_img.tif")))
    tifffile.imwrite(os.path.join(save_dir, "new_stitch_img.tif"), stitch_img, compression="jpeg")

    save_json(os.path.join(save_dir, "stitch_json.json"), stitch_json)

    return os.path.join(save_dir, "new_stitch_img.tif")


def cut_and_stitch(mrxs_path, reg_box):
    pic_dir = mrxs_path.replace(".mrxs", "-Cut")
    fov_shape, pic_shape = cut_fov_img(mrxs_path, pic_dir, conf.camera_resolution)
    print("End cut pic, start stitch pic...")
    save_dir = os.path.split(pic_dir)[0]
    stitch_pic_path = new_stitch(pic_dir, reg_box, pic_shape=pic_shape, save_dir=save_dir)

    return stitch_pic_path


if __name__ == '__main__':

    pic_dir = r"E:\new_stitch_test\cell-seg\20230303-BG13BN01F5-B3-FG3-20X-Cut"
    reg_box = [[4364, 4376], [29736, 4276], [29780, 29716], [4416, 29820]]
    pic_shape = (2048, 2448)
    save_dir = os.path.split(pic_dir)[0]
    _ = new_stitch(pic_dir, reg_box, pic_shape=pic_shape, save_dir=save_dir)
    # M = calculate_M(reg_box)
    # std_circle = StdCircles((30000, 30000), (30, 35), 17.565, 7)
    # stitch_img = np.zeros((30000, 30000, 3), dtype=np.uint8)
    #
    #
    # # whole_start_loc = reg_box[0]
    # # whole_end_loc = reg_box[2]
    # print("---end draw std circles")
    # print("start match img---")
    #
    # _, all_index_y_x = gen_index_by_reg_box(reg_box, [2048, 2448])
    # for index_y_x in all_index_y_x:
    #
    #     index_y, index_x = index_y_x
    #
    #     print("start match ori_{}_{}.tif".format(index_y, index_x))
    #     img_ori = cv2.imread(os.path.join(pics_dir, "ori_{}_{}.tif".format(index_y, index_x)))
    #     img = cv2.warpPerspective(img_ori, M, img_ori.shape[:2][::-1])
    #     out_img, centers = detector.detect(img, 0.4)
    #
    #     if not centers:
    #         print("Warnning: ori_{}_{}.tif has 0 Key Point, Skip it.".format(index_y, index_x))
    #         continue
    #
    #     max_match_kp = centers[0][:2]
    #
    #     # 计算max_match_kp在整个图像中的空间坐标
    #     max_match_kp_loc = [index_x * img_ori.shape[1] + max_match_kp[0], index_y * img_ori.shape[0] + max_match_kp[1]]
    #     # 计算max_match_kp相对芯片左上角的坐标
    #     max_match_kp_rel = np.asarray(max_match_kp_loc) - reg_box[0]
    #
    #     # 下面估算预测的关键点在std_mask底板中的位置
    #     distance = (np.asarray(reg_box[2]) - reg_box[0]) // 46
    #     tile_index_x, tile_index_y = np.round(max_match_kp_rel / distance).astype(int)
    #     kp_loc = std_circle.circle_centers[tile_index_x * 31 + 1, tile_index_y * 36 + 1]  # 找到对应块的第一个圆心坐标(x,y)，近似对应kp位置
    #     # show_img(std_circle.circles_mask[kp_loc[1]:kp_loc[1]+400, kp_loc[0]:kp_loc[0]+400])
    #
    #     # 从mask里面截取模板template，然后精准匹配
    #     template_start_loc = np.asarray(kp_loc) - max_match_kp
    #     template = std_circle.circles_mask[template_start_loc[1]-200:template_start_loc[1]+img_ori.shape[0]+200,
    #                                        template_start_loc[0]-200:template_start_loc[0]+img_ori.shape[1]+200]
    #
    #     match_result = cv2.matchTemplate(template, img[..., 2], cv2.TM_SQDIFF)
    #     match_shift = cv2.minMaxLoc(match_result)[2] - np.asarray([200, 200])
    #     real_loc = template_start_loc + match_shift
    #
    #     # 覆写mask
    #     # 先丢弃部分因仿射变换带来的黑边
    #     crop_rate = 0.002
    #     img_crop = img[int(img.shape[0] * crop_rate): -int(img.shape[0] * crop_rate),
    #                    int(img.shape[1] * crop_rate): -int(img.shape[1] * crop_rate)]
    #     real_loc_crop = real_loc + np.asarray([int(img.shape[1] * crop_rate), int(img.shape[0] * crop_rate)])
    #     std_circle.circles_mask[real_loc_crop[1]:real_loc_crop[1] + img_crop.shape[0],
    #                             real_loc_crop[0]:real_loc_crop[0] + img_crop.shape[1]] = img_crop[..., 2]
    #     stitch_img[real_loc_crop[1]:real_loc_crop[1] + img_crop.shape[0],
    #                real_loc_crop[0]:real_loc_crop[0] + img_crop.shape[1]] = img_crop
    #
    # save_dir = os.path.split(pics_dir)[0]
    # tifffile.imwrite(os.path.join(save_dir, r"\new_stitch_img_mask.tif"), std_circle.circles_mask, compression="jpeg")
    # tifffile.imwrite(os.path.join(save_dir, r"\new_stitch_img.tif"), stitch_img, compression="jpeg")
