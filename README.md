# QRCode_Axera
QRCode det & recognize DEMO on Axera
- 搜集二维码图片数据对轻量级目标检测模型进行默认参数训练，量化转换后统计板端模型性能及精度
- 目前支持ultralytics yolo/DEIMv2/NanodetPlus，提供yolov5/yolov8/DEIMv2/NanodetPlus系列二维码检测+zbar识别板端推理demo
- 目前支持 Python/C++ 语言 

## 支持平台

- [x] AX650N
- [x] AX630C
- [x] AX637

## 模型导出

YOLO系列参考[ultralytics](https://github.com/ultralytics/ultralytics) 中对模型导出方法，为方便部署去掉后处理部分，保留了三个输出分支，执行类似命令导出onnx模型：

```
yolo detect export model=yolov8n.pt format=onnx
```

DEIMv2系列参考[DEIMv2](https://github.com/wzf19947/DEIMv2) 中readme.txt模型导出方法，为方便部署去掉后处理部分，执行类似命令导出onnx模型：
```
python tools/deployment/export_onnx_deploy.py --check -c configs/deimv2/deimv2_hgnetv2_femto_coco.yml -r  weights/deimv2_hgnetv2_femto_coco.pth
```

NanoDet系列参考[NanoDet](https://github.com/wzf19947/Nanodet.git)中的模型导出方法
```
python export_onnx.py
```

## 模型转换

- Pulsar2 安装及使用请参考相关文档
  - [在线文档](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)

```
pulsar2 build --config ./yolo.json
or
pulsar2 build --config ./deimv2.json
or
pulsar2 build --config ./nanodet.json
```
得到对应模型用于部署的axmodel。
注:deimv2模型现阶段部署仅支持AX650上U16量化。

## 上板部署

- AX650N 的设备已预装 Ubuntu22.04
- 以 root 权限登陆 AX650N/AX630C/AX637 的板卡设备
- 已验证设备：AX650N/AX630C/AX637 DEMO Board

### Python API 运行

#### Requirements

二维码识别需要安装 pyzbar 库，安装命令如下:
```
sudo apt-get update
sudo apt-get install libzbar-dev
pip install pyzbar
``` 

#### 运行

##### 基于 ONNX Runtime 运行  
可在开发板或PC运行 

在开发板或PC上，运行以下命令  
```  
python3 QRCode_onnx_infer_xxx.py
```

##### 基于AXEngine运行  
在开发板上运行命令

```
python3 QRCode_axmodel_infer_xxx.py
```  


### 效果统计

#### AX650N

使用./qrcode_test下的图片作为测试集，进行检测+识别测试，效果统计如下：
![alt text](image.png)
```
注：
    1.外扩表示模型在检测到二维码后对检测框扩边，从原图截取对应区域后，再送至pyzbar库进行识别;检测模型后处理代码均从原工程中剥离，仅供参考，YOLOv8~v12模型均使用v8后处理逻辑；
    2.wechat_qrcode_opencv/opencv为二维码检测识别开源库，统计结果为直接输入原图测试。其余均为模型检测+crop+pyzbar识别结果；
    3.latency为模型推理耗时，整流程耗时大部分在preprocess和postprocess阶段。测试图片均为单二维码图片，耗时仅供参考；
```