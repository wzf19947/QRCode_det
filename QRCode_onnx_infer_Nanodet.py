import os
import glob
import time
import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
import onnxruntime as ort
import math

names = ["QRCode"]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def model_load(model):
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(model, providers=providers)
    input_name = session.get_inputs()[0].name
    output_names = [ x.name for x in session.get_outputs()]
    return session, output_names

def data_process_cv2(frame, input_shape):
    mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
    im0 = cv2.imread(frame)
    img = cv2.resize(im0, input_shape, interpolation=cv2.INTER_AREA).astype(np.float32)
    org_data = img.copy()
    img = (img - mean) / std
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    img = np.expand_dims(img, 0)
    return img, im0, org_data

def multiclass_nms(
    multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None
):
    num_classes = multi_scores.shape[1] - 1  # exclude background

    # Reshape bboxes
    if multi_bboxes.shape[1] > 4:
        # (N, 4*C) -> (N, C, 4)
        bboxes = multi_bboxes.reshape(multi_scores.shape[0], -1, 4)
    else:
        # (N, 4) -> (N, 1, 4) -> (N, C, 4) via repeat
        bboxes = np.tile(multi_bboxes[:, None, :], (1, num_classes, 1))

    scores = multi_scores[:, :-1].copy()  # (N, C)

    # Apply score factors if provided
    if score_factors is not None:
        scores = scores * score_factors[:, None]

    # Filter by score threshold
    valid_mask = scores > score_thr  # (N, C)

    # Get indices where valid
    valid_indices = np.where(valid_mask)
    if len(valid_indices[0]) == 0:
        # No valid boxes
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    # Extract valid bboxes, scores, labels
    bbox_indices, class_indices = valid_indices
    bboxes_valid = bboxes[bbox_indices, class_indices]  # (K, 4)
    scores_valid = scores[valid_indices]                # (K,)
    labels_valid = class_indices.astype(np.int64)       # (K,)

    # Concatenate bboxes and scores for NMS input: (K, 5)
    dets_input = np.concatenate([bboxes_valid, scores_valid[:, None]], axis=1)  # (K, 5)

    # Perform NMS (you need a NumPy NMS implementation)
    keep = nms_numpy(dets_input, iou_threshold=nms_cfg.get('iou_threshold', 0.5))

    dets = dets_input[keep]
    labels = labels_valid[keep]

    if max_num > 0 and len(keep) > max_num:
        dets = dets[:max_num]
        labels = labels[:max_num]

    return dets, labels
def nms_numpy(dets, iou_threshold=0.5):
    if dets.size == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # descending order

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep
def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop("class_agnostic", class_agnostic)

    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        # offsets = idxs * (max_coordinate + 1)
        offsets = idxs.astype(boxes.dtype) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

    nms_type = nms_cfg_.pop("type", "nms")  # unused in numpy version
    split_thr = nms_cfg_.pop("split_thr", 10000)

    if len(boxes_for_nms) < split_thr:
        # Call your NumPy NMS function (e.g., nms_numpy)
        keep = nms_numpy(boxes_for_nms, scores, **nms_cfg_)
        keep = np.array(keep, dtype=np.int64)
        boxes = boxes[keep]
        scores = scores[keep]
    else:
        # Large case: process per class/group
        total_mask = np.zeros(scores.shape, dtype=bool)
        unique_ids = np.unique(idxs)

        for id_val in unique_ids:
            mask = (idxs == id_val)
            mask_indices = np.where(mask)[0]  # indices where condition is True

            if len(mask_indices) == 0:
                continue

            keep_in_group = nms_numpy(
                boxes_for_nms[mask_indices], 
                scores[mask_indices], 
                **nms_cfg_
            )
            keep_in_group = np.array(keep_in_group, dtype=np.int64)
            selected_global_indices = mask_indices[keep_in_group]
            total_mask[selected_global_indices] = True

        keep = np.where(total_mask)[0]
        # Sort by scores descending
        sorted_indices = np.argsort(-scores[keep])  # negative for descending
        keep = keep[sorted_indices]
        boxes = boxes[keep]
        scores = scores[keep]

    # Concatenate boxes and scores -> (K, 5)
    dets = np.concatenate([boxes, scores[:, None]], axis=-1)
    return dets, keep

def scale_boxes_no_letter(img1_shape, boxes, img0_shape):
    gain = (img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])

    boxes[..., [0, 2]] /= gain[1]
    boxes[..., [1, 3]] /= gain[0]
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])

_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            0.000,
            0.447,
            0.741,
            0.314,
            0.717,
            0.741,
            0.50,
            0.5,
            0,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)

def distance2bbox(points, distance, max_shape=None):
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = np.clip(x1, a_min=0, a_max=max_shape[1])
        y1 = np.clip(y1, a_min=0, a_max=max_shape[0])
        x2 = np.clip(x2, a_min=0, a_max=max_shape[1])
        y2 = np.clip(y2, a_min=0, a_max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def integral_numpy(x, reg_max=16):
    """
    NumPy equivalent of the Integral layer in NanoDet.
    
    Computes: sum(softmax(logits) * [0, 1, ..., reg_max]) for each of the 4 directions.
    
    Args:
        x (np.ndarray): Input array of shape (..., 4 * (reg_max + 1))
        reg_max (int): Maximum value of discrete set. Default: 16.
    
    Returns:
        np.ndarray: Integral result of shape (..., 4)
    """
    # Save original leading shape (e.g., (N,) or (N, H, W))
    leading_shape = x.shape[:-1]  # everything except last dim
    total_channels = x.shape[-1]
    
    assert total_channels == 4 * (reg_max + 1), \
        f"Last dimension must be 4*(reg_max+1)={4*(reg_max+1)}, but got {total_channels}"
    
    # Reshape to (..., 4, reg_max + 1)
    x = x.reshape(*leading_shape, 4, reg_max + 1)
    
    # Apply softmax along the last axis (dim=-1)
    # For numerical stability: subtract max
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    softmax_x = exp_x / np.sum(exp_x, axis=-1, keepdims=True)  # (..., 4, reg_max+1)
    
    # Project vector: [0, 1, 2, ..., reg_max]
    project = np.arange(reg_max + 1, dtype=x.dtype)  # shape (reg_max+1,)
    
    # Compute weighted sum: sum(softmax_x * project) over last dimension
    # Broadcasting: (..., 4, reg_max+1) * (reg_max+1,) -> (..., 4, reg_max+1)
    integral_result = np.sum(softmax_x * project, axis=-1)  # (..., 4)
    
    return integral_result

def overlay_bbox_cv(img, dets, class_names, score_thresh):
    all_box = []
    for label in dets:
        for bbox in dets[label]:
            score = bbox[-1]
            if score > score_thresh:
                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                all_box.append([label, x0, y0, x1, y1, score])
    all_box.sort(key=lambda v: v[5])
    # for box in all_box:
    #     label, x0, y0, x1, y1, score = box
    #     # color = self.cmap(i)[:3]
    #     color = (_COLORS[label] * 255).astype(np.uint8).tolist()
    #     text = "{}:{:.1f}%".format(class_names[label], score * 100)
    #     txt_color = (0, 0, 0) if np.mean(_COLORS[label]) > 0.5 else (255, 255, 255)
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]
        # cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        # cv2.rectangle(
        #     img,
        #     (x0, y0 - txt_size[1] - 1),
        #     (x0 + txt_size[0] + txt_size[1], y0 - 1),
        #     color,
        #     -1,
        # )
        # cv2.putText(img, text, (x0, y0 - 1), font, 0.5, txt_color, thickness=1)
    return img, all_box
    
class NanoDetONNXInfer:
    def __init__(self, model_path, imgsz=[416, 416]):
        self.model_path = model_path
        self.session, self.output_names = model_load(self.model_path)
        self.imgsz = imgsz
        self.reg_max = 7
        self.reg_max1= self.reg_max + 1
        self.distribution_project = np.arange(self.reg_max + 1)
        self.nc = len(names)
        self.no = self.nc + self.reg_max1 * 4
        self.stride = [8, 16, 32, 64]

    def get_bboxes(self, cls_preds, reg_preds):
        """Decode the outputs to bboxes.
        Args:
            cls_preds (Tensor): Shape (num_imgs, num_points, num_classes).
            reg_preds (Tensor): Shape (num_imgs, num_points, 4 * (regmax + 1)).
            img_metas (dict): Dict of image info.

        Returns:
            results_list (list[tuple]): List of detection bboxes and labels.
        """
        b = cls_preds.shape[0]

        featmap_sizes = [
            (math.ceil(self.imgsz[0] / stride), math.ceil(self.imgsz[1]) / stride)
            for stride in self.stride
        ]

        # get grid cells of one image
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                b,
                featmap_sizes[i],
                stride,
                dtype=np.float32,
            )
            for i, stride in enumerate(self.stride)
        ]

        center_priors = np.concatenate(mlvl_center_priors, axis=1)
        integral_result = integral_numpy(reg_preds, reg_max=self.reg_max)  # (N, 4)
        scale = center_priors[..., 2][..., None]  # shape (N, 1) or (N, H, W, 1)
        dis_preds = integral_result * scale
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=self.imgsz)
        scores = 1.0 / (1.0 + np.exp(-cls_preds))  # sigmoid
        result_list = []
        for i in range(b):
            # add a dummy background class at the end of all labels
            # same with mmdetection2.0
            score, bbox = scores[i], bboxes[i]
            padding = np.zeros((score.shape[0], 1), dtype=score.dtype)
            score = np.concatenate([score, padding], axis=1)
            results = multiclass_nms(
                bbox,
                score,
                score_thr=0.05,
                nms_cfg=dict(type="nms", iou_threshold=0.6),
                max_num=100,
            )
            result_list.append(results)
        return result_list
    def get_single_level_center_priors(self,batch_size, featmap_size, stride, dtype):
        h, w = featmap_size
        x_range = (np.arange(w, dtype=dtype)) * stride
        y_range = (np.arange(h, dtype=dtype)) * stride
        y, x = np.meshgrid(y_range, x_range, indexing='ij')
        y = y.flatten()
        x = x.flatten()
        strides = np.full((x.shape[0],), stride, dtype=dtype)
        priors = np.stack([x, y, strides, strides], axis=-1)
        return np.tile(priors[None, :, :], (batch_size, 1, 1))

    def detect_objects(self, image, save_path):
        outputs=[]
        im, im0, org_data = data_process_cv2(image, self.imgsz)
        img_name = os.path.basename(image).split('.')[0]
        infer_start_time = time.time()
        x = self.session.run(None, {self.session.get_inputs()[0].name: im})
        infer_end_time = time.time()
        print(f"infer time: {infer_end_time - infer_start_time:.4f}s")
        x = [np.transpose(x[i],(0,3,1,2)) for i in range(4)]    #to nchw
        for i in range(len(x)):
            reg_pred = x[i][:, :self.reg_max1 * 4,:,:]
            cls_pred = x[i][:, self.reg_max1 * 4:,:,:]
            out = np.concatenate([cls_pred, reg_pred], axis=1)
            outputs.append(out.reshape(out.shape[0], out.shape[1], -1))
        preds = np.concatenate(outputs, axis=2).transpose(0, 2, 1)

        cls_scores = preds[:, :, :self.nc]
        bbox_preds = preds[:, :, self.nc:]
        pred = self.get_bboxes(cls_scores, bbox_preds)[0]
        res = self.post_process(pred, org_data, im0, save_path, img_name)
        result_img, bbox_res = overlay_bbox_cv(im0, res, names, score_thresh=0.35)
        return bbox_res, result_img
    def post_process(self, result, im, im0, save_path, img_name):
        det_result = {}
        det_bboxes, det_labels = result
        det_bboxes[:, :4] = scale_boxes_no_letter(im.shape[:2], det_bboxes[:, :4], im0.shape).round()
        classes = det_labels
        for i in range(self.nc):
            inds = classes == i
            det_result[i] = np.concatenate(
                [
                    det_bboxes[inds, :4].astype(np.float32),
                    det_bboxes[inds, 4:5].astype(np.float32),
                ],
                axis=1,
            ).tolist()

        return det_result
    
class QRCodeDecoder:
    def crop_qr_regions(self, image, regions):
        """
        根据检测到的边界框裁剪二维码区域
        """
        cropped_images = []
        for idx, region in enumerate(regions):
            label, x1, y1, x2, y2, score = region
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

    detector = NanoDetONNXInfer(model_path='./nanodet-plus-m_416_QR.onnx',imgsz=[416,416])
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
        # print('det_result:',det_result)
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