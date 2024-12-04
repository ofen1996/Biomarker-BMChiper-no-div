import os

import cv2
import tifffile

vipsbin = r'D:\work\python\vips-dev-8.15\bin'
os.environ['PATH'] = r"./src/vips-dev-8.15/bin" + ';' + vipsbin + ';' + os.environ['PATH']
import pyvips
import xml.etree.ElementTree as ET


def rotate_and_cinter_crop(im, angle):
    return im.rotate(angle).smartcrop(im.width, im.height, interesting="centre")

def read_pyramid_from_file(*args, **kwargs):
    return pyvips.Image.new_from_file(*args, **kwargs)

def read_pyramid_from_array(*args, **kwargs):
    return pyvips.Image.new_from_array(*args, **kwargs)

def read_3_page_as_bands(pyramid_img_path, *args, **kwargs):
    whole_img = read_pyramid_from_file(pyramid_img_path, page=0, *args, **kwargs)
    # G_B_channel = []
    # for i in (1, 2):
    #     tmp = read_pyramid_from_file(pyramid_img_path, page=i)
    #     G_B_channel.append(tmp)
    # whole_img = whole_img
    whole_img = whole_img.bandjoin([read_pyramid_from_file(pyramid_img_path, page=channel, *args, **kwargs) for channel in (1, 2)])
    return whole_img


def draw_black(width, height, bands=1):
    return pyvips.Image.black(width, height, bands=bands)


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
                pyramid=True, properties=True, bigtiff=True)


def get_property_value(im, property_name):
    xml_string = im.get("image-description")
    return get_property_value_from_xml(xml_string, property_name)


def get_property_value_from_xml(xml_string, property_name):
    """
    从XML字符串中提取给定property名称的值。

    参数:
        xml_string (str): 包含XML数据的字符串。
        property_name (str): 要查询的property的名称。

    返回:
        str: 对应property的值。如果找不到，返回None。
    """
    # 移除命名空间声明
    cleaned_xml_string = xml_string.replace('xmlns="http://www.vips.ecs.soton.ac.uk//dzsave"', '')

    # 解析 XML 字符串
    root = ET.fromstring(cleaned_xml_string)

    # 找到所有 property 元素
    properties = root.find('properties')
    if properties is not None:
        for prop in properties.findall('property'):
            name = prop.find('name').text
            if name == property_name:
                return prop.find('value').text

    # 如果没有找到对应的property，返回None
    return None


def save_special_size(ome_tif_path, save_path, new_weight, new_height, compression="lzw"):
    target_height = new_height
    # 加载顶层图像以获取 SubIFD 信息
    image = pyvips.Image.new_from_file(ome_tif_path)

    # 获取 SubIFD 层数
    num_levels = image.get("n-subifds")  # 顶层也算一个层级

    # 遍历 SubIFD 层，找到最接近目标高度的层
    closest_layer = -1
    closest_height = float("inf")

    for level in range(num_levels):
        layer_image = pyvips.Image.new_from_file(ome_tif_path, subifd=level)
        height = layer_image.height  # 使用高度作为判断条件
        print(f"sub images height: {height}")

        if height >= target_height and height < closest_height:
            closest_layer = level
            closest_height = height

    # 如果找到合适的层，读取并保存
    if closest_layer != -1:
        print(f"选择的subifds金字塔层: {closest_layer}, 高度: {closest_height}")
        selected_layer_image = read_3_page_as_bands(ome_tif_path, subifd=closest_layer)
    else:
        print("未找到符合要求的金字塔层。选择顶层操作")
        selected_layer_image = read_3_page_as_bands(ome_tif_path, memory=True)  # 此处memory=True是为了接近重复读取时候，报奇怪的错

    img_dist = selected_layer_image.numpy()
    img_dist = cv2.resize(img_dist, (new_weight, new_height))
    tifffile.imwrite(save_path, img_dist, compression=compression)
    print(f"保存完成：{save_path}")