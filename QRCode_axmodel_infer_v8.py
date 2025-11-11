import cv2
import numpy as np
import time
import yaml
import glob
import os
import pyzbar.pyzbar as pyzbar
import axengine as axe

names=['QRCode']

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
    img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
    img = np.asarray(img, dtype=np.uint8)
    img = np.expand_dims(img, 0)
    # img /= 255.0
    return img, im0, org_data

def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        labels=(),
                        max_det=300,
                        nm=0  # number of masks
                        ):
    """
    Perform Non-Maximum Suppression (NMS) on the boxes to filter out overlapping boxes.

    Parameters:
    prediction (ndarray): Predictions from the model.
    conf_thres (float): Confidence threshold to filter boxes.
    iou_thres (float): Intersection over Union (IoU) threshold for NMS.
    classes (list): Filter boxes by classes.
    agnostic (bool): If True, perform class-agnostic NMS.
    multi_label (bool): If True, perform multi-label NMS.
    labels (list): Labels for auto-labelling.
    max_det (int): Maximum number of detections.
    nm (int): Number of masks.

    Returns:
    list: A list of filtered boxes.
    """
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    # redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    # merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [np.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros((len(lb), nc + nm + 5))
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[np.arange(len(lb)), lb[:, 0].astype(int) + 5] = 1.0  # cls
            x = np.concatenate((x, v), 0)

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
            i, j = np.nonzero(x[:, 5:mi] > conf_thres)
            x = np.concatenate((box[i], x[i, 5 + j][:, None], j[:, None].astype(float), mask[i]), 1)
        else:  # best class only
            # conf = x[:, 5:mi].max(1, keepdims=True)
            # j = x[:, 5:mi].argmax(1,keepdims=True)
            conf = np.max(x[:, 5:mi], 1).reshape(box.shape[:1][0], 1)
            j = np.argmax(x[:, 5:mi], 1).reshape(box.shape[:1][0], 1)
            x = np.concatenate((box, conf, j.astype(float), mask), 1)[conf[:, 0] > conf_thres]
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)[:, None]).any(1)]
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        sorted_indices = np.argsort(x[:, 4])[::-1]
        x = x[sorted_indices][:max_nms]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms(boxes, scores, iou_thres)  # NMS

        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        # if mps:
        #     output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            # LOGGER.warning(f'WARNING  NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded
    return output


# Define the function for NMS using numpy
def nms(boxes, scores, iou_threshold):
    """
    Perform Non-Maximum Suppression (NMS) on the given boxes with scores using numpy.

    Parameters:
    boxes (ndarray): The bounding boxes, shaped (N, 4).
    scores (ndarray): The confidence scores for each box, shaped (N,).
    iou_threshold (float): The IoU threshold for suppressing overlapping boxes.

    Returns:
    ndarray: The indices of the selected boxes after NMS.
    """
    if len(boxes) == 0:
        return []

    # Sort boxes by their scores
    indices = np.argsort(scores)[::-1]

    selected_indices = []
    while len(indices) > 0:
        # Select the box with the highest score
        current_index = indices[0]
        selected_indices.append(current_index)

        # Compute IoU between the current box and all other boxes
        current_box = boxes[current_index]
        other_boxes = boxes[indices[1:]]
        iou = calculate_iou(current_box, other_boxes)

        # Remove boxes with IoU higher than the threshold
        indices = indices[1:][iou <= iou_threshold]

    return np.array(selected_indices)


def calculate_iou(box, boxes):
    """
    Calculate the Intersection over Union (IoU) between a given box and a set of boxes.

    Parameters:
    box (ndarray): The coordinates of the first box, shaped (4,).
    boxes (ndarray): The coordinates of the other boxes, shaped (N, 4).

    Returns:
    ndarray: The IoU between the given box and each box in the set, shaped (N,).
    """
    # Calculate intersection coordinates
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    # Calculate intersection area
    intersection_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    # Calculate areas of both bounding boxes
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Calculate IoU
    iou = intersection_area / (box_area + boxes_area - intersection_area)

    return iou

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
    if len(det):
        det[:, :4] = scale_boxes(im.shape[:2], det[:, :4], im0.shape).round()
        colors = Colors()
        detections = []
        for *xyxy, conf, cls in reversed(det):
            # print("class:",int(cls), "left:%.0f" % xyxy[0],"top:%.0f" % xyxy[1],"right:%.0f" % xyxy[2],"bottom:%.0f" % xyxy[3], "conf:",'{:.0f}%'.format(float(conf)*100))
            int_coords = [int(tensor.item()) for tensor in xyxy]
            # print(int_coords)
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
    providers = ['CPUExecutionProvider']
    # session = ort.InferenceSession(model, providers=providers)
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
        _, _, h, w = feats[i].shape
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
        self.reg_max = 16
        self.nc = 1
        self.no = self.nc + self.reg_max * 4
        self.dfl = DFL(self.reg_max)

    def detect_objects(self, image, save_path):
        im, im0, org_data = data_process_cv2(image, self.imgsz)
        img_name = os.path.basename(image).split('.')[0]
        infer_start_time = time.time()
        x = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        infer_end_time = time.time()
        print(f"infer time: {infer_end_time - infer_start_time:.4f}s")
        x = [np.transpose(x[i],(0,3,1,2)) for i in range(3)]    #to nchw

        anchors,strides = (np.transpose(x,(1, 0)) for x in make_anchors(x, self.stride, 0.5))
        x_cat = np.concatenate([xi.reshape(1, self.no, -1) for xi in x], axis=2)
        box = x_cat[:, :self.reg_max * 4,:]
        cls = x_cat[:, self.reg_max * 4:,:]
        dbox = dist2bbox(self.dfl(box), np.expand_dims(anchors, axis=0), xywh=True, dim=1) * strides
        y = np.concatenate((dbox, 1/(1 + np.exp(-cls))), axis=1)
        pred = y.transpose([0, 2, 1])
        pred_class = pred[..., 4:]
        pred_conf = np.max(pred_class, axis=-1)
        pred = np.insert(pred, 4, pred_conf, axis=-1)

        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)
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

    detector = YOLOV8Detector(model_path='./yolov8n_npu3.axmodel',imgsz=[640,640])
    decoder = QRCodeDecoder()
    img_path = './qrcode_test'
    det_path='./v8_det_res'
    crop_path='./v8_crop_res'
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
            print("识别成功！")
        else:
            fail += 1
            print("识别失败！")
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