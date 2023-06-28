import configparser
import os


class Config:
    def __init__(self, ini_path):
        self.ini_path = ini_path
        self.conf = None

        self.base_mode = "S1000"
        self.base_size_x = 46
        self.base_size_y = 46

        self.DEBUG = None
        self.use_real_kp_only = None
        self.kp_detect_confidence = None
        self.kp_loc_confidence = None
        self.shift_y = None
        self.shift_x = None
        self.compression_mode = None
        self.std_mask_color = None
        self.out_size = None
        self.calculate_size = None

        self.std_edge_size = None
        self.whole_img_size = None
        self.stitch_channal = None

        self.overlap_x = None
        self.overlap_y = None
        self.x_range = None
        self.y_range = None
        self.resize_level = None
        self.camera_resolution = None
        self.if_raise_exception = None

        self.load(self.ini_path)

    def reload(self):
        self.load(self.ini_path)

    def load(self, ini_path):
        self.conf = configparser.ConfigParser()
        self.conf.read(ini_path, encoding="utf-8")

        self.base_mode = self.conf.get("default", "base_mode")
        if self.base_mode == "S2000":
            self.base_size_x = 76
            self.base_size_y = 75

        self.calculate_size = int(self.conf.get("correct-whole-img", "calculate_size"))
        self.out_size = int(self.conf.get("correct-whole-img", "max_out_size"))
        self.std_mask_color = eval(self.conf.get("correct-whole-img", "std_mask_color"))
        self.compression_mode = eval(self.conf.get("correct-whole-img", "compression_mode"))
        self.shift_x = float(self.conf.get("correct-whole-img", "shift_x"))
        self.shift_y = float(self.conf.get("correct-whole-img", "shift_y"))
        self.use_real_kp_only = eval(self.conf.get("correct-whole-img", "use_real_kp_only"))
        self.kp_detect_confidence = float(self.conf.get("correct-whole-img", "kp_detect_confidence"))
        self.kp_loc_confidence = float(self.conf.get("correct-whole-img", "kp_loc_confidence"))
        self.DEBUG = eval(self.conf.get("correct-whole-img", "DEBUG"))

        self.std_edge_size = eval(self.conf.get("match-imgs", "std_edge_size"))
        self.whole_img_size = eval(self.conf.get("match-imgs", "whole_img_size"))
        self.stitch_channal = int(self.conf.get("match-imgs", "stitch_channal"))

        self.overlap_x = int(self.conf.get("stitch", "overlap_x"))
        self.overlap_y = int(self.conf.get("stitch", "overlap_y"))
        self.x_range = int(self.conf.get("stitch", "x_range"))
        self.y_range = int(self.conf.get("stitch", "y_range"))
        self.resize_level = int(self.conf.get("stitch", "resize_level"))
        self.camera_resolution = eval(self.conf.get("fov-cut", "camera_resolution"))
        self.if_raise_exception = eval(self.conf.get("fov-cut", "if_raise_exception"))


if os.path.exists("./setting/setting.ini"):
    conf = Config("./setting/setting.ini")
else:
    conf = Config("../setting/setting.ini")
