import numpy as np
import cv2
import os
import torch
import time
import torchvision
import matplotlib
import pyzbar.pyzbar as pyzbar
import axengine as axe

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        self.palette = [self.hex2rgb(c) for c in matplotlib.colors.TABLEAU_COLORS.values()]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

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
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
        list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
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
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
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
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, kpt_label=False, step=2):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]
    if isinstance(gain, (list, tuple)):
        gain = gain[0]
    if not kpt_label:
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, [0, 2]] /= gain
        coords[:, [1, 3]] /= gain
        clip_coords(coords[0:4], img0_shape)
        #coords[:, 0:4] = coords[:, 0:4].round()
    else:
        coords[:, 0::step] -= pad[0]  # x padding
        coords[:, 1::step] -= pad[1]  # y padding
        coords[:, 0::step] /= gain
        coords[:, 1::step] /= gain
        clip_coords(coords, img0_shape, step=step)
        #coords = coords.round()
    return coords


def clip_coords(boxes, img_shape, step=2):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0::step].clamp_(0, img_shape[1])  # x1
    boxes[:, 1::step].clamp_(0, img_shape[0])  # y1



def plot_one_box(x, im, color=None, label=None, line_thickness=3, steps=2, orig_shape=None):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl*1//3, lineType=cv2.LINE_AA)
    if label:
        if len(label.split(' ')) > 1:
            # label = label.split(' ')[-1]
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 6, [225, 255, 255], thickness=tf//2, lineType=cv2.LINE_AA)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



class Yolov5QRcodeDetector:
    def __init__(self, model_path):
        # self.model = onnxruntime.InferenceSession(model_path)
        self.model = axe.InferenceSession(model_path)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        self.classes=['QRCode']
        self.nc=len(self.classes)
        self.no = self.nc + 5
        self.na =3
        self.nl =3
        self.anchors=torch.tensor([[10,13, 16,30, 33,23],[30,61, 62,45, 59,119],[116,90, 156,198, 373,326]])
        self.anchors=self.anchors.view(3,3,2)
        self.stride=torch.tensor([8,16,32])
        self.anchors = self.anchors/(self.stride.view(-1, 1, 1))

    def preprocess_image(self, img, img_size=(640, 640)):
        img, _, _ = letterbox(img, img_size, auto=False, stride=32)
        img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
        # img = np.asarray(img, dtype=np.float32)
        img = np.asarray(img, dtype=np.uint8)
        img = np.expand_dims(img, 0)
        # img /= 255.0
        return img

    def model_inference(self, input=None):
        output = self.model.run(None, {self.input_name: input})
        return output

    def _make_grid(self, nx=20, ny=20, i=0):
        na = 3
        shape = 1, na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, dtype=torch.float32), torch.arange(nx, dtype=torch.float32)
        # yv, xv = torch.meshgrid(y, x)  # torch>=0.7 compatibility
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid
    def postprocess(self, preds, img_shape, im0):
        z = []  # inference output
        for i,pred in enumerate(preds):
            pred=torch.from_numpy(pred)   #numpy2tensor
            pred=pred.permute(0,3,1,2)  #NHWC to NCHW
            bs, _, ny, nx = pred.shape
            pred = pred.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            grid, anchor_grid = self._make_grid(nx, ny, i)

            xy, wh, conf = sigmoid(pred).split((2, 2, self.nc + 1), 4)
            xy = (xy * 2 + grid) * self.stride[i]  # xy
            wh = (wh * 2) ** 2 * anchor_grid  # wh
            y = torch.cat((xy, wh, conf), 4)
            z.append(y.view(bs, self.na * nx * ny, self.no))

        preds=torch.cat(z, 1)
        detections = []
        preds = non_max_suppression(preds, 0.3, 0.45)
        for i, det in enumerate(preds):  # detections per image

            if len(det):
                # Rescale boxes from img_size to im0 size
                scale_coords(img_shape[2:], det[:, :4], im0.shape, kpt_label=False)

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class

                # Write results
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    # print('det:',xyxy, conf, cls)
                    int_coords = [int(tensor.item()) for tensor in xyxy]
                    # print(int_coords)
                    detections.append(int_coords)
                    # c = int(cls)  # integer class
                    # label =  f'{self.classes[c]} {conf:.2f}'
                    # plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=2,steps=3, orig_shape=im0.shape[:2])

        return detections, im0

class QRCodeDecoder:
    def crop_qr_regions(self, image, regions):
        """
        根据检测到的边界框裁剪二维码区域
        """
        cropped_images = []
        for idx, region in enumerate(regions):
            x1, y1, x2, y2 = region
            # 外扩缓解检测截断，视检测情况而定
            x1-=15
            y1-=15
            x2+=15
            y2+=15
            # 裁剪图像
            cropped = image[y1:y2, x1:x2]
            if cropped.size > 0:
                cropped_images.append({
                    'image': cropped,
                    'bbox': region,
                })
                # cv2.imwrite(f'cropped_qr_{idx}.jpg', cropped)
        return cropped_images

    def decode_qrcode_pyzbar(self, cropped_image):
        """
        使用pyzbar解码二维码
        """
        try:
            # 转换为灰度图像
            if len(cropped_image.shape) == 3:
                gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = cropped_image
            # cv2.imwrite('cropped_gray.jpg',gray)
            # 使用pyzbar解码
            decoded_objects = pyzbar.decode(gray)
            results = []
            for obj in decoded_objects:
                try:
                    data = obj.data.decode('utf-8')
                    results.append({
                        'data': data,
                        'type': obj.type,
                        'points': obj.polygon
                    })
                except:
                    continue
            
            return results
        except Exception as e:
            print(f"decode error: {e}")
            return []


if __name__ == '__main__':
    import time

    model = './yolov5n_npu3.axmodel'
    input_size = [640,640]
    detector = Yolov5QRcodeDetector(model)
    # Crop deteted QRCode & decode QRCode by pyzbar
    decoder = QRCodeDecoder()
    pic_path = './qrcode_test/'
    det_path='./v5_det_res'
    crop_path='./v5_crop_res'
    os.makedirs(det_path, exist_ok=True)
    os.makedirs(crop_path, exist_ok=True)
    pics = os.listdir(pic_path)
    totoal = len(pics)
    success = 0
    fail = 0
    start_time = time.time()  # 记录总开始时间
    for idx, pic in enumerate(pics):
        loop_start_time = time.time()  # 记录单张图片开始时间
        org_img = os.path.join(pic_path, pic)
        pic_name=pic.split('.')[0]
        im0 = cv2.imread(org_img)

        #do QRCode detection
        img = detector.preprocess_image(im0, img_size=input_size)
        infer_start_time = time.time()
        preds = detector.model_inference(img)
        infer_end_time = time.time()
        print(f"infer time: {infer_end_time - infer_start_time:.4f}s")
        det_result, res_img = detector.postprocess(preds, img.shape, im0)
        # cv2.imwrite(os.path.join(det_path, pic), res_img)

        cropped_images = decoder.crop_qr_regions(im0, det_result)
        for i,cropped in enumerate(cropped_images):
            cv2.imwrite(os.path.join(crop_path, f'{pic_name}_crop_{i}.jpg'), cropped['image'])
        
        all_decoded_results = []
        for i, cropped_data in enumerate(cropped_images):
            decoded_results = decoder.decode_qrcode_pyzbar(cropped_data['image'])
            all_decoded_results.extend(decoded_results)
            
            # for result in decoded_results:
            #     print(f"decode result: {result['data']} (type: {result['type']})")
        if all_decoded_results:
            success += 1
            # print("识别成功！")
        else:
            fail += 1
            # print("识别失败！")
        loop_end_time = time.time()  # 记录单张图片结束时间
        print(f"图片 {pic} 处理耗时: {loop_end_time - loop_start_time:.4f} 秒")

    end_time = time.time()  # 记录总结束时间
    total_time = end_time - start_time  # 记录总耗时

    print(f"总共测试图片数量: {totoal}")
    print(f"识别成功数量: {success}")
    print(f"识别失败数量: {fail}")
    print(f"识别成功率: {success/totoal*100:.2f}%")
    print(f"整体处理耗时: {total_time:.4f} 秒")
    print(f"平均每张图片处理耗时: {total_time/totoal:.4f} 秒")
