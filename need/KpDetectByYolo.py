import os, sys
import time
import argparse
import torch
import torchvision
import cv2
from need.ofen_tool import show_img, save_json

DETECT_TYPE = ["kp"]


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # # Rescale boxes (xyxy) from img1_shape to img0_shape
    # if ratio_pad is None:  # calculate from img0_shape
    #     gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    #     pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    # else:
    #     gain = ratio_pad[0][0]
    #     pad = ratio_pad[1]
    #
    # boxes[:, [0, 2]] -= pad[0]  # x padding
    # boxes[:, [1, 3]] -= pad[1]  # y padding
    # boxes[:, :4] /= gain

    boxes[:, [0, 2]] /= (img1_shape[1] / img0_shape[1])
    boxes[:, [1, 3]] /= (img1_shape[0] / img0_shape[0])

    clip_boxes(boxes, img0_shape)
    return boxes


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    prediction = torch.asarray(prediction.tolist())

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    # output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    output = [torch.zeros(0, 6 + nm)] * bs

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #     # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        # if mps:
        #     output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            # LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            print(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


def draw_boxes(img, boxes):
    centers = []
    for box in boxes:  # box = [x1, y1, x2, y2, conf, type]
        color = (0, 255, 0)
        line_width = max(round(sum(img.shape) / 2 * 0.003), 2)
        x1, y1, x2, y2, conf, t_type = box.tolist()
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, line_width)
        cv2.circle(img, (round((x1 + x2)/2), round((y1 + y2)/2)), 3, (0, 0, 0), -1)
        cv2.putText(img, DETECT_TYPE[int(t_type)] + str(round(conf, 2)), (int(x1), int(y1)), 0, line_width/3, color,
                    thickness=max(line_width - 1, 1))
        centers.append([round((x1 + x2)/2), round((y1 + y2)/2), conf])
    return centers

class MyDetector:
    def __init__(self, weight_path):
        self.net = cv2.dnn.readNetFromONNX(weight_path)
        self.input_shape = (640, 640)

    def detect(self, img, conf_thres=0.5):
        t = time.time()
        img = img.copy()
########### 此段代码仅在零时使用，后续删除###############
        img = cv2.merge([img[..., 2], img[..., 0], img[..., 1]])
###############################################
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, self.input_shape, swapRB=True, crop=False)
        self.net.setInput(blob)
        pred = self.net.forward()
        pred_nms = non_max_suppression(pred, conf_thres)

        # Process predictions
        centers = []
        for i, det in enumerate(pred_nms):
            if len(det):
                det[:, :4] = scale_boxes(self.input_shape, det[:, :4], img.shape).round()
                centers = draw_boxes(img, det)

        return img, centers


if __name__ == '__main__':
    print(sys.argv)
    parser = argparse.ArgumentParser(description="Find Key-Points in S1000")
    parser.add_argument('--out', type=str, help="result save dir")
    parser.add_argument('--source', type=str, help="img file path or dir")
    parser.add_argument('--weight', type=str, help="weight path '.onnx'", required=True)
    parser.add_argument('--conf', type=float, help="conf")

    args = parser.parse_args()
    print(args)

    source = args.source
    weight_path = args.weight
    save_dir = args.out
    conf_thres = args.conf

    detector = MyDetector(weight_path)

    if os.path.isdir(source):
        pass
    else:
        img_name = os.path.split(source)[1]
        img = cv2.imread(source)
        out_img, centers = detector.detect(img, conf_thres)
        if save_dir:
            cv2.imwrite(os.path.join(save_dir, "detect_{}".format(img_name)), out_img)
            save_json(os.path.join(save_dir, os.path.splitext(img_name)[0] + ".json"), centers)
        else:
            show_img(out_img)
