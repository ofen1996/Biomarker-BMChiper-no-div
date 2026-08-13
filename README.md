# BMChiper

面向 BioMarker 空间芯片图像的桌面处理工具。项目提供芯片区域识别、图像校正、超大图拼接与多通道合并能力，针对 `.mrxs`、`.svs`、OME-TIFF/Zarr 等全切片或大尺寸图像工作流设计。

> 本仓库为内部工具代码，模型、参数和数据格式与具体扫描仪/芯片平台相关。

## 能力概览

- 使用 ONNX 关键点检测模型定位芯片阵列与有效区域
- 读取并预览 MRXS、SVS、TIFF 与 Zarr 图像
- 拼接大尺寸图像与多通道图像
- 借助 libvips 降低超大图处理的内存压力
- 根据 `setting/` 中的预设适配 20X、40X 及不同扫描仪
- 提供 PyQt5 图形界面与 PyInstaller 打包配置

## 快速开始

建议在 Windows + Python 3.9 环境中运行：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy pillow opencv-python matplotlib PyQt5 onnxruntime tifffile pyvips scikit-learn
python BMChiper.py
```

处理 MRXS 文件需要安装 OpenSlide；使用 libvips 脚本时还需安装系统级 libvips，并保证其动态库可被系统找到。

## 使用流程

1. 运行 `BMChiper.py` 启动图形界面。
2. 打开原始图像或图像目录，选择与采集参数匹配的 `setting/*.ini` 配置。
3. 在界面中完成芯片区域确认、整体校正或拼接。
4. 对多通道数据，可使用 `merge_channels_pics_by_libvips.py` 或 `stitch_channels_pics_by_libvips.py`。

默认配置文件为 `setting/setting.ini`；改动配置后请确认其中的扫描仪类型、放大倍率、图块尺寸与输入数据一致。

## 主要文件

| 路径 | 说明 |
| --- | --- |
| `BMChiper.py` | PyQt5 桌面应用入口 |
| `match_imgs.py` | 芯片阵列匹配、关键点定位与图像配准 |
| `stitch_pic.py` | 图像拼接流程 |
| `merge_*_by_libvips.py` | 面向大图的 libvips 合并脚本 |
| `need/` | 图像读取、区域 UI、校正与工具模块 |
| `model/*.onnx` | 芯片关键点检测模型 |
| `setting/` | 设备与芯片规格预设 |

## 输出与注意事项

处理过程中会生成 TIFF、JSON、日志和中间结果；请使用有充足磁盘空间的工作目录。大图像处理会消耗大量内存，建议先使用小数据验证设置。项目中的路径及部分 libvips 配置以 Windows 部署为主，跨平台运行时可能需要调整。

## 许可与支持

本仓库未声明开源许可证。未经维护方书面许可，请勿将模型、业务数据或内部部署配置用于仓库授权范围之外的用途。
