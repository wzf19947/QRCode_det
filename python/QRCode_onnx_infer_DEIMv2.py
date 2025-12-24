"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""
import os
import sys
import cv2
import time
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw
import pyzbar.pyzbar as pyzbar
import torchvision
import glob

def mod(a, b):
    out = a - a // b * b
    return out

mscoco_category2name = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}

class PostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'remap_mscoco_category'
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)

            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        if self.deploy_mode:
            return labels, boxes, scores

        if self.remap_mscoco_category:
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
def resize_with_aspect_ratio(image, size, interpolation=Image.BILINEAR):
    """Resizes an image while maintaining aspect ratio and pads it."""
    original_width, original_height = image.size
    ratio = min(size / original_width, size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    image = image.resize((new_width, new_height), interpolation)

    # Create a new image with the desired size and paste the resized image onto it
    new_image = Image.new("RGB", (size, size))
    new_image.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
    return new_image, ratio, (size - new_width) // 2, (size - new_height) // 2


def draw(images, labels, boxes, scores, ratios, paddings, thrh=0.25):
    result_images = []
    detections=[]
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scr = scr[scr > thrh]

        ratio = ratios[i]
        pad_w, pad_h = paddings[i]

        for lbl, bb in zip(lab, box):
            # Adjust bounding boxes according to the resizing and padding
            bb = [
                (bb[0] - pad_w) / ratio,
                (bb[1] - pad_h) / ratio,
                (bb[2] - pad_w) / ratio,
                (bb[3] - pad_h) / ratio,
            ]
            draw.rectangle(bb, outline='red')
            draw.text((bb[0], bb[1]), text=str(lbl), fill='blue')
            detection=[int(bb[i]) for i in range(len(bb))]
            detections.append(detection)
        result_images.append(im)
    return result_images, detections


def process_image(sess, im_pil, post_processor, size=640, model_size='s'):

    # Resize image while preserving aspect ratio
    resized_im_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(im_pil, size)
    orig_size = torch.tensor([[resized_im_pil.size[1], resized_im_pil.size[0]]])

    transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
                if model_size not in ['atto', 'femto', 'pico', 'n'] 
                else T.Lambda(lambda x: x)
        ])
    im_data = transforms(resized_im_pil).unsqueeze(0)

    output = sess.run(
        output_names=None,
        input_feed={'images': im_data.numpy()}
    )

    output = {"pred_logits": torch.from_numpy(output[0]), "pred_boxes": torch.from_numpy(output[1])}
    output=post_processor(output,orig_size)
    labels, boxes, scores = output

    result_images, detections = draw(
        [im_pil], labels, boxes, scores,
        [ratio], [(pad_w, pad_h)]
    )

    return detections, result_images


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

    #load the ONNX model
    sess = ort.InferenceSession('deimv2_hgnetv2_femto_coco.onnx')
    size = sess.get_inputs()[0].shape[2]
    print(f"Using device: {ort.get_device()}")

    #QRCode decoder
    decoder = QRCodeDecoder()
    img_path = './qrcode_test'
    det_path='./DEIMv2_det_res'
    crop_path='./DEIMv2_crop_res'

    os.makedirs(det_path, exist_ok=True)
    os.makedirs(crop_path, exist_ok=True)
    #get post info from your trained model config
    post_processor = PostProcessor(use_focal_loss=True, num_classes=1, num_top_queries=100)
    post_processor.deploy()
    # print('post_processor:',post_processor)
    imgs = glob.glob(f"{img_path}/*.jpg")
    totoal = len(imgs)
    success = 0
    fail = 0
    start_time = time.time()
    for idx,img in enumerate(imgs):
        pic_name=os.path.basename(img).split('.')[0]
        loop_start_time = time.time()
        #detect image
        im_pil = Image.open(img).convert('RGB')
        img_cv2 = np.array(im_pil)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
        det_result, res_img = process_image(sess, im_pil, post_processor, size, 'femto')
        res_img[0].save(os.path.join(det_path, pic_name+'.jpg'))

        # Crop deteted QRCode & decode QRCode by pyzbar
        cropped_images = decoder.crop_qr_regions(img_cv2, det_result)
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
