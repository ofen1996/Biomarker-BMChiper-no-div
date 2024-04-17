import configparser
import os


class Config:
    def __init__(self, ini_path):
        self.ini_path = ini_path
        self.conf = None
        self.machine = None

        self.base_mode = "S1000"
        self.base_size_x = 46
        self.base_size_y = 46

        self.barcode_size_x = 30
        self.barcode_size_y = 35
        self.mrxs_read_level = 0

        self.std_d = "auto"

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
        # self.load(self.ini_path)
        self.__init__(self.ini_path)

    def load(self, ini_path):
        self.conf = configparser.ConfigParser()
        self.conf.read(ini_path, encoding="utf-8")

        self.machine = self.conf.get("default", "machine")

        if self.conf.has_option("default", "mrxs_read_level"):
            self.mrxs_read_level = int(self.conf.get("default", "mrxs_read_level"))
        if self.conf.has_option("default", "std_d"):
            self.std_d = self.conf.get("default", "std_d")

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

        self.base_mode = self.conf.get("default", "base_mode")
        if self.base_mode == "S2000":
            self.base_size_x = 76
            self.base_size_y = 75
        elif self.base_mode == "S1000":
            self.base_size_x = 46
            self.base_size_y = 46
        elif self.base_mode == "S2000-2":
            self.base_size_x = 101
            self.base_size_y = 134
            self.out_size = 30000
        elif self.base_mode == "S3000":
            self.barcode_size_x = 45
            self.barcode_size_y = 51
            self.base_size_x = 42
            self.base_size_y = 43



if os.path.exists("./setting/setting.ini"):
    conf = Config("./setting/setting.ini")
else:
    conf = Config("../setting/setting.ini")
