# QRCode_C++ demo
QRCode det & recognize C++ DEMO on Axera
- 提供yolov5/yolov8/DEIMv2/NanodetPlus系列二维码检测+zbar识别板端C++推理demo

## 支持平台

- [x] AX650N
- [x] AX630C
- [x] AX637


## 模型转换

- Pulsar2 安装及使用请参考相关文档
  - [在线文档](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)

```
pulsar2 build --config ./yolo_cpp.json
or
pulsar2 build --config ./deimv2_cpp.json
or
pulsar2 build --config ./nanodet.json

得到对应模型用于部署的axmodel。
注:deimv2模型现阶段部署仅支持AX650上U16量化。
```

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

以AX650为例，其编译器为aarch64-none-linux-gnu，则执行命令：
```
cd zbar-master/
autoreconf -vfi
./configure --host=aarch64-none-linux-gnu -prefix=$PWD/build_ax650
make clean && make & make install
```
AX630C、AX637编译方法同上，更换对应的编译器执行命令即可。

4.拷贝库和头文件到SDK相应目录
```
cd build_ax650/
scp lib/libzbar.a /home/workspace/Feng/ax-samples-main/ax650n_bsp_sdk-main/msp/out/lib/
scp lib/libzbar.so* /home/workspace/Feng/ax-samples-main/ax650n_bsp_sdk-main/msp/out/lib/
scp -r include/* /home/workspace/Feng/ax-samples-main/ax650n_bsp_sdk-main/msp/out/include/
```

5.编译上板demo

开源项目[AX_Samples](https://github.com/AXERA-TECH/ax-samples)实现了常见的深度学习开源算法在 爱芯元智 的 AI SoC 上的示例代码，方便社区开发者进行快速评估和适配。
最新版本已开始提供 AX650 系列（AX650A、AX650N）、AX620E 系列（AX630C、AX620Q）的 NPU 示例。

以AX650为例，编译参考[compile_650.md](https://github.com/AXERA-TECH/ax-samples/blob/main/docs/compile_650.md), 编译.cc文件得到可执行程序ax_yolov8_qrcode_batch、ax_yolov5_qrcode_batch、ax_deimv2_qrcode_batch、ax_nanodetplus_qrcode_batch。

AX630C、AX637的板端编译方法参考对应的compile_xxx.md即可。


#### 运行

##### 基于AXEngine运行  
将编好的动态库文件拷贝到开发板：

```
cd build_ax650/
scp lib/libzbar.so* root@10.126.XX.1XX:/opt/lib/
```

将所需可执行文件、模型、图片等拷贝到开发板，并在开发板上运行命令：

```
./ax_xxx_qrcode_batch -m ./yolov8n_cpp_npu3.axmodel -i ./qrcode_test/
./ax_yolov5_qrcode_batch -m ./yolov5n_cpp_npu3.axmodel -i ./qrcode_test/
./ax_deimv2_qrcode_batch -m ./deimv2_hgnetv2_femto_coco_cpp_npu3.axmodel -i ./qrcode_test/
./ax_nanodetplus_qrcode_batch -m nanodet-plus-m_416_QR.axmodel - i qrcode_test/
```  

### 板端结果
```
使用./qrcode_test下的图片作为测试集，进行检测+识别测试，效果如下：
--------------------------------------
image path: qrcode_test//qrcode_55.jpg image index: qrcode_55
post process cost time:0.93 ms
--------------------------------------
Repeat 1 times, avg time 3.73 ms, max_time 3.73 ms, min_time 3.73 ms
--------------------------------------
detection num: 1
 0:  96%, [1021,  749, 1145,  869], QRCode
ZBAR cut region = [123 x 119]
ZBAR scan n = 0
ZBAR scan success use expand size of 153x150
Decode data:[C:CNS:Aixin-GuestP:aixinguestK:e1QbyyUv], type:[QR-Code]
--------------------------------------
Total pics:48
Total decode count:42
Decode rate:87.5%

```