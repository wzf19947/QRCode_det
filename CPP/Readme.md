# QRCode_C++ demo
QRCode det & recognize C++ DEMO on Axera
- 提供yolov5/yolov8二维码检测+zbar识别板端C++推理demo
- 目前支持 C++ 语言 

## 支持平台

- [x] AX650N

## 模型转换

- Pulsar2 安装及使用请参考相关文档
  - [在线文档](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)

```
pulsar2 build --config ./yolo_cpp.json
```
得到对应模型用于部署的axmodel。

## C++ demo编译

#### Requirements

二维码识别需要安装 zbar 库，交叉编译方法如下:
1.下载源码：
```
git clone https://github.com/mchehab/zbar.git
```

2.安装依赖库
```
sudo apt-get install autoconf autopoint pkg-config libtool gcc make gettext libpng-dev
```

3.交叉编译库
```
cd zbar-master/
autoreconf -vfi
./configure --host=aarch64-none-linux-gnu -prefix=$PWD/build_ax650
make clean && make & make install
```

4.拷贝库和头文件到SDK相应目录
```
cd build_ax650/
scp lib/libzbar.a /home/workspace/Feng/ax-samples-main/ax650n_bsp_sdk-main/msp/out/lib/
scp lib/libzbar.so* /home/workspace/Feng/ax-samples-main/ax650n_bsp_sdk-main/msp/out/lib/
scp -r include/* /home/workspace/Feng/ax-samples-main/ax650n_bsp_sdk-main/msp/out/include/
```

5.编译上板demo
开源项目 AX-Samples 实现了常见的深度学习开源算法在 爱芯元智 的 AI SoC 上的示例代码，方便社区开发者进行快速评估和适配。
最新版本已开始提供 AX650 系列（AX650A、AX650N）、AX620E 系列（AX630C、AX620Q）的 NPU 示例，其中也包含了本文介绍的 YOLO_Uniow 参考代码。
[AX_Samples](https://github.com/AXERA-TECH/ax-samples)

编译参考./compile_650.md, 编译ax_yolov8_qrcode_batch.cc、ax_yolov5_qrcode_batch.cc得到可执行文件ax_yolov8_qrcode_batch、ax_yolov5_qrcode_batch


#### 运行

##### 基于AXEngine运行  
将所需可执行文件、模型、图片等拷贝到开发板，并在开发板上运行命令：

```
./ax_yolov8_qrcode_batch -m ./yolov8n_cpp_npu3.axmodel -i ./qrcode_test/ -o ./res
./ax_yolov5_qrcode_batch -m ./yolov5n_cpp_npu3.axmodel -i ./qrcode_test/ -o ./res
```  

### 板端结果

#### AX650N
```
使用./qrcode_test下的图片作为测试集，进行检测+识别测试，效果统计如下：
image path: ./qrcode_test//qrcode_54.jpg image index: qrcode_54
post process cost time:0.92 ms
--------------------------------------
Repeat 1 times, avg time 1.41 ms, max_time 1.41 ms, min_time 1.41 ms
--------------------------------------
detection num: 1
 0:  84%, [1013,  709, 1135,  829], QRCode
ZBAR cut region = [121 x 119]
ZBAR scan n = 0
--------------------------------------
image path: ./qrcode_test//qrcode_55.jpg image index: qrcode_55
post process cost time:0.97 ms
--------------------------------------
Repeat 1 times, avg time 1.42 ms, max_time 1.42 ms, min_time 1.42 ms
--------------------------------------
detection num: 1
 0:  83%, [1020,  749, 1148,  869], QRCode
ZBAR cut region = [128 x 120]a
ZBAR scan n = 0
ZBAR scan success use ThresholdType=3 thr=157
Decode data:[C:CNS:Aixin-GuestP:aixinguestK:e1QbyyUv], type:[QR-Code]
--------------------------------------
Total pics:50
Total decode count:39
Decode rate:78.0%
```