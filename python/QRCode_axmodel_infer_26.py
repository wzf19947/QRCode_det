import axengine as axe
import cv2
import numpy as np
import time
import yaml
import glob
import os
import torch
from pyzbar import pyzbar
names = [
    "QRCode"
]

def non_max_suppression(
    prediction,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels=(),
    max_det: int = 300,
    nc: int = 0,  # number of classes (optional)
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    rotated: bool = False,
    end2end: bool = False,
    return_idxs: bool = False,
):
    """Perform non-maximum suppression (NMS) on prediction results.

    Applies NMS to filter overlapping bounding boxes based on confidence and IoU thresholds. Supports multiple detection
    formats including standard boxes, rotated boxes, and masks.

    Args:
        prediction (torch.Tensor): Predictions with shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing boxes, classes, and optional masks.
        conf_thres (float): Confidence threshold for filtering detections. Valid values are between 0.0 and 1.0.
        iou_thres (float): IoU threshold for NMS filtering. Valid values are between 0.0 and 1.0.
        classes (list[int], optional): List of class indices to consider. If None, all classes are considered.
        agnostic (bool): Whether to perform class-agnostic NMS.
        multi_label (bool): Whether each box can have multiple labels.
        labels (list[list[Union[int, float, torch.Tensor]]]): A priori labels for each image.
        max_det (int): Maximum number of detections to keep per image.
        nc (int): Number of classes. Indices after this are considered masks.
        max_time_img (float): Maximum time in seconds for processing one image.
        max_nms (int): Maximum number of boxes for NMS.
        max_wh (int): Maximum box width and height in pixels.
        rotated (bool): Whether to handle Oriented Bounding Boxes (OBB).
        end2end (bool): Whether the model is end-to-end and doesn't require NMS.
        return_idxs (bool): Whether to return the indices of kept detections.

    Returns:
        output (list[torch.Tensor]): List of detections per image with shape (num_boxes, 6 + num_masks) containing (x1,
            y1, x2, y2, confidence, class, mask1, mask2, ...).
        keepi (list[torch.Tensor]): Indices of kept detections if return_idxs=True.
    """
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6 or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    extra = prediction.shape[1] - nc - 4  # number of extra info
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
    xinds = torch.arange(prediction.shape[-1], device=prediction.device).expand(bs, -1)[..., None]  # to track idxs

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
    keepi = [torch.zeros((0, 1), device=prediction.device)] * bs  # to store the kept idxs
    for xi, (x, xk) in enumerate(zip(prediction, xinds)):  # image index, (preds, preds indices)
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        filt = xc[xi]  # confidence
        x = x[filt]
        if return_idxs:
            xk = xk[filt]

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + extra + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, extra), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            if return_idxs:
                xk = xk[i]
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            filt = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float(), mask), 1)[filt]
            if return_idxs:
                xk = xk[filt]

        # Filter by class
        if classes is not None:
            filt = (x[:, 5:6] == classes).any(1)
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            filt = x[:, 4].argsort(descending=True)[:max_nms]  # sort by confidence and remove excess boxes
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_probiou)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            # Speed strategy: torchvision for val or already loaded (faster), TorchNMS for predict (lower latency)
            if "torchvision" in sys.modules:
                import torchvision  # scope as slow import

                i = torchvision.ops.nms(boxes, scores, iou_thres)
            else:
                i = TorchNMS.nms(boxes, scores, iou_thres)
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        if return_idxs:
            keepi[xi] = xk[i].view(-1)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return (output, keepi) if return_idxs else output

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    
    shape = im.shape[:2]  
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  
        r = min(r, 1.0)
 
    ratio = r, r  
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  
    if auto:  
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  
    elif scaleFill:  
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  
 
    dw /= 2  
    dh /= 2
 
    if shape[::-1] != new_unpad:  
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  
    return im, ratio, (dw, dh)

def data_process_cv2(frame, input_shape):
    '''
    对输入的图像进行预处理
    :param frame:
    :param input_shape:
    :return:
    '''
    im0 = cv2.imread(frame)
    img = letterbox(im0, input_shape, auto=False, stride=32)[0]
    org_data = img.copy()
    img = np.ascontiguousarray(img[:, :, ::-1])
    img = np.asarray(img, dtype=np.uint8)
    img = np.expand_dims(img, 0)
    return img, im0, org_data

# Define xywh2xyxy function for converting bounding box format
def xywh2xyxy(x):
    """
    Convert bounding boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2) format.

    Parameters:
    x (ndarray): Bounding boxes in (center_x, center_y, width, height) format, shaped (N, 4).

    Returns:
    ndarray: Bounding boxes in (x1, y1, x2, y2) format, shaped (N, 4).
    """
    y = x.copy()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def post_process_yolo(det, im, im0, gn, save_path, img_name):
    detections = []
    if len(det):
        det[:, :4] = scale_boxes(im.shape[:2], det[:, :4], im0.shape).round()
        colors = Colors()
        for *xyxy, conf, cls in reversed(det):
            print("class:",int(cls), "left:%.0f" % xyxy[0],"top:%.0f" % xyxy[1],"right:%.0f" % xyxy[2],"bottom:%.0f" % xyxy[3], "conf:",'{:.0f}%'.format(float(conf)*100))
            int_coords = [int(tensor.item()) for tensor in xyxy]
            detections.append(int_coords)
            # c = int(cls)
            # label = names[c]
            # res_img = plot_one_box(xyxy, im0, label=f'{label}:{conf:.2f}', color=colors(c, True), line_thickness=4)
            # cv2.imwrite(f'{save_path}/{img_name}.jpg',res_img)
            # xywh = (xyxy2xywh(np.array(xyxy,dtype=np.float32).reshape(1, 4)) / gn).reshape(-1).tolist()  # normalized xywh
            # line = (cls, *xywh)  # label format
            # with open(f'{save_path}/{img_name}.txt', 'a') as f:
            #     f.write(('%g ' * len(line)).rstrip() % line + '\n')   
    return detections

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])


def yaml_load(file='coco128.yaml'):
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        """
        Initializes the Colors class with a palette derived from Ultralytics color scheme, converting hex codes to RGB.
        Colors derived from `hex = matplotlib.colors.TABLEAU_COLORS.values()`.
        """
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        """Returns color from palette by index `i`, in BGR format if `bgr=True`, else RGB; `i` is an integer index."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))

def plot_one_box(x, im, color=None, label=None, line_thickness=3, steps=2, orig_shape=None):
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl*1//3, lineType=cv2.LINE_AA)
    if label:
        if len(label.split(':')) > 1:
            tf = max(tl - 1, 1)  
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 6, [225, 255, 255], thickness=tf//2, lineType=cv2.LINE_AA)
    return im

def model_load(model):
    session = axe.InferenceSession(model)
    input_name = session.get_inputs()[0].name
    output_names = [ x.name for x in session.get_outputs()]
    return session, output_names

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype = feats[0].dtype
    for i, stride in enumerate(strides):
        # _, _, h, w = feats[i].shape
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = np.arange(w, dtype=dtype) + grid_cell_offset  # shift x
        sy = np.arange(h, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = np.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(np.stack((sx, sy), axis=-1).reshape(-1, 2))
        stride_tensor.append(np.full((h * w, 1), stride, dtype=dtype))
    return np.concatenate(anchor_points), np.concatenate(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = np.split(distance, 2, axis=dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return np.concatenate((c_xy, wh), axis=dim)  # xywh bbox
    return np.concatenate((x1y1, x2y2), axis=dim)  # xyxy bbox


class DFL:
    """
    NumPy implementation of Distribution Focal Loss (DFL) integral module.
    Original paper: Generalized Focal Loss (IEEE TPAMI 2023)
    """
    
    def __init__(self, c1=16):
        """Initialize with given number of distribution channels"""
        self.c1 = c1
        # 初始化权重矩阵（等效于原conv层的固定权重）
        self.weights = np.arange(c1, dtype=np.float32).reshape(1, c1, 1, 1)
        

    def __call__(self, x):
        """
        前向传播逻辑
        参数:
            x: 输入张量，形状为(batch, channels, anchors)
        返回:
            处理后的张量，形状为(batch, 4, anchors)
        """
        b, c, a = x.shape
        
        # 等效于原view->transpose->softmax操作
        x_reshaped = x.reshape(b, 4, self.c1, a)
        x_transposed = np.transpose(x_reshaped, (0, 2, 1, 3))
        x_softmax = np.exp(x_transposed) / np.sum(np.exp(x_transposed), axis=1, keepdims=True)
        
        # 等效卷积操作(通过张量乘积实现)
        conv_result = np.sum(self.weights * x_softmax, axis=1)
        
        return conv_result.reshape(b, 4, a)
    
class YOLOV8Detector:
    def __init__(self, model_path, imgsz=[640,640]):
        self.model_path = model_path
        self.session, self.output_names = model_load(self.model_path)
        self.imgsz = imgsz
        self.stride = [8.,16.,32.]
        self.reg_max = 1
        self.nc = len(names)
        self.nl = len(self.stride)
        self.dfl = DFL(self.reg_max)
        self.max_det = 300

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        boxes, scores = preds.split([4, self.nc], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores, conf], dim=-1)

    def get_topk_index(self, scores: torch.Tensor, max_det: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get top-k indices from scores.

        Args:
            scores (torch.Tensor): Scores tensor with shape (batch_size, num_anchors, num_classes).
            max_det (int): Maximum detections per image.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): Top scores, class indices, and filtered indices.
        """
        batch_size, anchors, nc = scores.shape  # i.e. shape(1,8400,84)
        # Use max_det directly during export for TensorRT compatibility (requires k to be constant),
        # otherwise use min(max_det, anchors) for safety with small inputs during Python inference
        k = max_det
        #对8400个anchor取其80类中的最大类概率，shape[1,8400]--再取topk，shape[1,k]--unsqueeze,shape[1,k,1]
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        #[1,k,1]repeat变为[1,k,80]，从[1,8400,80]中取topk个完整logit
        scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        #展平从k*80个分数中取topk。总体就是先删选topk个最可能anchor，再从该anchor中取topk个最可能class
        scores, index = scores.flatten(1).topk(k)
        #映射回原位置
        idx = ori_index[torch.arange(batch_size)[..., None], index // nc]  # original index
        return scores[..., None], (index % nc)[..., None].float(), idx
    
    def detect_objects(self, image, save_path):
        im, im0, org_data = data_process_cv2(image, self.imgsz)
        img_name = os.path.basename(image).split('.')[0]
        infer_start_time = time.time()
        x = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        infer_end_time = time.time()
        print(f"infer time: {infer_end_time - infer_start_time:.4f}s")
        x = [np.transpose(x[i],(0,3,1,2)) for i in range(self.nl)]    #to nchw
        anchors,strides = (np.transpose(x,(1, 0)) for x in make_anchors(x, self.stride, 0.5))
        box = [x[i][:, :self.reg_max * 4,:] for i in range(self.nl)]
        cls = [x[i][:, self.reg_max * 4:,:] for i in range(self.nl)]
        boxes = np.concatenate([box[i].reshape(1, 4 * self.reg_max, -1) for i in range(self.nl)], axis=-1)
        scores = np.concatenate([cls[i].reshape(1, self.nc, -1) for i in range(self.nl)], axis=-1)
        if self.reg_max > 1:
            dbox = dist2bbox(self.dfl(boxes), np.expand_dims(anchors, axis=0), xywh=False, dim=1) * strides
        else:   #弃用DFL
            dbox = dist2bbox(boxes, np.expand_dims(anchors, axis=0), xywh=False, dim=1) * strides
        y = np.concatenate((dbox, 1/(1 + np.exp(-scores))), axis=1)
        y = y.transpose([0, 2, 1])
        pred = self.postprocess(torch.from_numpy(y))
        pred = non_max_suppression(
            pred.cpu().numpy(),
            0.25,
            0.7,
            None,
            False,
            max_det=300,
            nc=0,
            end2end=True,
            rotated=False,
            return_idxs=None,
        )
        gn = np.array(org_data.shape)[[1, 0, 1, 0]].astype(np.float32)
        res = post_process_yolo(pred[0], org_data, im0, gn, save_path, img_name)
        return res, im0

class QRCodeDecoder:
    def crop_qr_regions(self, image, regions):
        """
        根据检测到的边界框裁剪二维码区域
        """
        cropped_images = []
        for idx, region in enumerate(regions):
            x1, y1, x2, y2 = region
            # 外扩15个像素缓解因检测截断造成无法识别的情况，视检测情况而定
            # x1-=15
            # y1-=15
            # x2+=15
            # y2+=15
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

    detector = YOLOV8Detector(model_path='./yolo26n.axmodel',imgsz=[640,640])
    decoder = QRCodeDecoder()
    img_path = './qrcode_test'
    det_path='./det_res'
    crop_path='./crop_res'
    os.makedirs(det_path, exist_ok=True)
    os.makedirs(crop_path, exist_ok=True)
    imgs = glob.glob(f"{img_path}/*.jpg")
    totoal = len(imgs)
    success = 0
    fail = 0
    start_time = time.time()
    for idx,img in enumerate(imgs):
        pic_name=os.path.basename(img).split('.')[0]
        loop_start_time = time.time()
        det_result, res_img = detector.detect_objects(img,det_path)
        # cv2.imwrite(os.path.join(det_path, pic_name+'.jpg'), res_img)

        # Crop deteted QRCode & decode QRCode by pyzbar
        cropped_images = decoder.crop_qr_regions(res_img, det_result)
        # for i,cropped in enumerate(cropped_images):
        #     cv2.imwrite(os.path.join(crop_path, f'{pic_name}_crop_{i}.jpg'), cropped['image'])

        all_decoded_results = []
        for i, cropped_data in enumerate(cropped_images):
            decoded_results = decoder.decode_qrcode_pyzbar(cropped_data['image'])
            all_decoded_results.extend(decoded_results)
            
            # for result in decoded_results:
            #     print(f"decode result: {result['data']} (type: {result['type']})")
        if all_decoded_results:
            success += 1
            print(f"{pic_name} 识别成功！")
        else:
            fail += 1
            print(f"{pic_name} 识别失败！")
        loop_end_time = time.time()
        print(f"图片 {img} 处理耗时: {loop_end_time - loop_start_time:.4f} 秒")

    end_time = time.time()  # 记录总结束时间
    total_time = end_time - start_time  # 记录总耗时

    print(f"总共测试图片数量: {totoal}")
    print(f"识别成功数量: {success}")
    print(f"识别失败数量: {fail}")
    print(f"识别成功率: {success/totoal*100:.2f}%")
    print(f"整体处理耗时: {total_time:.4f} 秒")
    print(f"平均每张图片处理耗时: {total_time/totoal:.4f} 秒")