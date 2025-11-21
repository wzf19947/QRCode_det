/*
* AXERA is pleased to support the open source community by making ax-samples available.
*
* Copyright (c) 2022, AXERA Semiconductor (Shanghai) Co., Ltd. All rights reserved.
*
* Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
* in compliance with the License. You may obtain a copy of the License at
*
* https://opensource.org/licenses/BSD-3-Clause
*
* Unless required by applicable law or agreed to in writing, software distributed
* under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
* CONDITIONS OF ANY KIND, either express or implied. See the License for the
* specific language governing permissions and limitations under the License.
*/

/*
* Author: ZHEQIUSHUI
*/

#include <cstdio>
#include <cstring>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "base/common.hpp"
#include "base/detection.hpp"
#include "middleware/io.hpp"

#include "utilities/args.hpp"
#include "utilities/cmdline.hpp"
#include "utilities/file.hpp"
#include "utilities/timer.hpp"

#include <ax_sys_api.h>
#include <ax_engine_api.h>
#include <zbar.h>

const int DEFAULT_IMG_H = 640;
const int DEFAULT_IMG_W = 640;
const int ZBAR_DECODE_H = 192;
const int ZBAR_DECODE_W = 192;

const char* CLASS_NAMES[] = {
    "QRCode"};
const float ANCHORS[18] = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};
int NUM_CLASS = 1;

const int DEFAULT_LOOP_COUNT = 1;

const float PROB_THRESHOLD = 0.45f;
const float NMS_THRESHOLD = 0.45f;
namespace ax
{
    std::vector<detection::Object> post_process(AX_ENGINE_IO_INFO_T* io_info, AX_ENGINE_IO_T* io_data, const cv::Mat& mat, int input_w, int input_h, const std::vector<float>& time_costs, std::string output_dir, std::string basename)
    {
        std::vector<detection::Object> proposals;
        std::vector<detection::Object> objects;
        float prob_threshold_u_sigmoid = -1.0f * (float)std::log((1.0f / PROB_THRESHOLD) - 1.0f);
        timer timer_postprocess;
        for (uint32_t i = 0; i < io_info->nOutputSize; ++i)
        {
            auto& output = io_data->pOutputs[i];
            auto& info = io_info->pOutputs[i];
            auto ptr = (float*)output.pVirAddr;
            int32_t stride = (1 << i) * 8;
            detection::generate_proposals_yolov5(stride, ptr, PROB_THRESHOLD, proposals, input_w, input_h, ANCHORS, prob_threshold_u_sigmoid,NUM_CLASS);
        }

        detection::get_out_bbox(proposals, objects, NMS_THRESHOLD, input_h, input_w, mat.rows, mat.cols);
        fprintf(stdout, "post process cost time:%.2f ms \n", timer_postprocess.cost());
        fprintf(stdout, "--------------------------------------\n");
        auto total_time = std::accumulate(time_costs.begin(), time_costs.end(), 0.f);
        auto min_max_time = std::minmax_element(time_costs.begin(), time_costs.end());
        fprintf(stdout,
                "Repeat %d times, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n",
                (int)time_costs.size(),
                total_time / (float)time_costs.size(),
                *min_max_time.second,
                *min_max_time.first);
        fprintf(stdout, "--------------------------------------\n");
        fprintf(stdout, "detection num: %zu\n", objects.size());
        // std::string output_img_name = output_dir + "/" + basename;
        // std::string output_txt_name = output_dir + "/" + basename;
        // detection::draw_objects(mat, objects, CLASS_NAMES, output_img_name.c_str());
        // detection::save_txt(mat, objects, output_txt_name);
        // for (size_t i = 0; i < objects.size(); i++)
        // {
        //     detection::Object obj = objects[i];
        //     fprintf(stdout, "%2d: %3.0f%%, [%4.0f, %4.0f, %4.0f, %4.0f], %s\n", obj.label, obj.prob * 100, obj.rect.x,
        //             obj.rect.y, obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, CLASS_NAMES[obj.label]);
        //     cv::Mat roi_image = mat(obj.rect);
        //     cv::imwrite("zbar.jpg", roi_image);
        // }
        return objects;
    }

    bool run_model(const std::string& model, std::string images_dir, const int& repeat, int input_h, int input_w, std::string output_dir)
    {
        // 1. init engine
        AX_ENGINE_NPU_ATTR_T npu_attr;
        memset(&npu_attr, 0, sizeof(npu_attr));
        npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
        auto ret = AX_ENGINE_Init(&npu_attr);
        if (0 != ret)
        {
            return ret;
        }

        // 2. load model
        std::vector<char> model_buffer;
        if (!utilities::read_file(model, model_buffer))
        {
            fprintf(stderr, "Read Run-Joint model(%s) file failed.\n", model.c_str());
            return false;
        }

        // 3. create handle
        AX_ENGINE_HANDLE handle;
        ret = AX_ENGINE_CreateHandle(&handle, model_buffer.data(), model_buffer.size());
        SAMPLE_AX_ENGINE_DEAL_HANDLE
        fprintf(stdout, "Engine creating handle is done.\n");

        // 4. create context
        ret = AX_ENGINE_CreateContext(handle);
        SAMPLE_AX_ENGINE_DEAL_HANDLE
        fprintf(stdout, "Engine creating context is done.\n");

        // 5. set io
        AX_ENGINE_IO_INFO_T* io_info;
        ret = AX_ENGINE_GetIOInfo(handle, &io_info);
        SAMPLE_AX_ENGINE_DEAL_HANDLE
        fprintf(stdout, "Engine get io info is done. \n");

        // 6. alloc io
        AX_ENGINE_IO_T io_data;
        ret = middleware::prepare_io(io_info, &io_data, std::make_pair(AX_ENGINE_ABST_DEFAULT, AX_ENGINE_ABST_CACHED));
        SAMPLE_AX_ENGINE_DEAL_HANDLE
        fprintf(stdout, "Engine alloc io is done. \n");

        // 7. insert input
        // 读取路径内图片列表, 以jpg图片为例
        std::string surffix = "*.jpg";
        std::vector<std::string> files_vector;
        utilities::file_list(images_dir, surffix, files_vector);

        //zbar init
        zbar::zbar_image_scanner_t *scanner = NULL;
        scanner = zbar::zbar_image_scanner_create();
        zbar::zbar_image_scanner_set_config(scanner, zbar::ZBAR_QRCODE, zbar::ZBAR_CFG_ENABLE, 1);
        zbar::zbar_image_t *zbarimage = zbar::zbar_image_create();
        zbar::zbar_image_set_format(zbarimage, zbar_fourcc('Y', '8', '0', '0'));

        int total_decode_count = 0;
        for (int index = 0 ; index < files_vector.size(); index++)
        {
            bool success = false;
            std::string file_name = files_vector[index];
            std::string image_path = images_dir + "/" + file_name;
            std::string basename = file_name.substr(0, file_name.rfind("."));
            printf("image path: %s image index: %s\n", image_path.c_str(), basename.c_str());
            std::vector<uint8_t> image(input_h * input_w * 3, 0);
            cv::Mat mat = cv::imread(image_path);
            if (mat.empty())
            {
                fprintf(stderr, "Read image failed.\n");
                return -1;
            }
            cv::Mat image_org = mat.clone();
            common::get_input_data_letterbox(mat, image, input_h, input_w, true);
        
            ret = middleware::push_input(image, &io_data, io_info);
            if (0 != ret)
            {
                printf("middleware::push_input error !!!\n");
                continue;
            }
            // SAMPLE_AX_ENGINE_DEAL_HANDLE_IO
            // fprintf(stdout, "Engine push input is done. \n");
            // fprintf(stdout, "--------------------------------------\n");

        // 8. warn up
            // for (int i = 0; i < 5; ++i)
            // {
            //     AX_ENGINE_RunSync(handle, &io_data);
            // }

        // 9. run model
        std::vector<float> time_costs(repeat, 0);
        for (int i = 0; i < repeat; ++i)
        {
            timer tick;
            ret = AX_ENGINE_RunSync(handle, &io_data);
            time_costs[i] = tick.cost();
            SAMPLE_AX_ENGINE_DEAL_HANDLE_IO
        }

        // 10. get result
            std::vector<detection::Object> QR_Regions = post_process(io_info, &io_data, mat, input_w, input_h, time_costs, output_dir, basename);
            for (size_t i = 0; i < QR_Regions.size(); i++)
            {
                detection::Object obj = QR_Regions[i];
                fprintf(stdout, "%2d: %3.0f%%, [%4.0f, %4.0f, %4.0f, %4.0f], %s\n", obj.label, obj.prob * 100, obj.rect.x,
                        obj.rect.y, obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, CLASS_NAMES[obj.label]);
                cv::Mat roi_image = image_org(obj.rect);
                int cut_width  = (int)obj.rect.width;
                int cut_height = (int)obj.rect.height;
                fprintf(stdout,"ZBAR cut region = [%d x %d]\n", cut_width,cut_height);
                cv::Mat gray;
                cv::cvtColor(roi_image, gray, cv::COLOR_BGR2GRAY);
                zbar::zbar_image_set_size(zbarimage, cut_width, cut_height);
                zbar::zbar_image_set_data(zbarimage, gray.data, cut_width * cut_height, NULL);
                int n = zbar_scan_image(scanner, zbarimage);
                fprintf(stdout,"ZBAR scan n = %d\n", n);
                
                // 若初次未扫描成功，可添加图像处理、缩放等策略提高扫码成功率
                // 二值化
                if (n < 1) {
                    cv::ThresholdTypes func[2] = {cv::THRESH_BINARY, cv::THRESH_TOZERO};
                    for (int retry = 0; retry < 2; retry++) {
                        for (int thr = 97; thr <= 157; thr += 15) {
                            cv::Mat srcImage = gray.clone();
                            cv::Mat dstImage;
                            threshold(srcImage, dstImage, thr, 255, func[retry]);
                            zbar::zbar_image_set_data(zbarimage, dstImage.data, cut_width * cut_height, NULL);;
                            n = zbar_scan_image(scanner, zbarimage);
                            if (n > 0) {
                                fprintf(stdout, "ZBAR scan success use ThresholdType=%d thr=%d\n", func[retry], thr);
                                goto AFTER_SCAN;
                            } else {
                                dstImage.release();
                            }
                        }
                    }
                }

                //高斯模糊
                if (n < 1) {
                    cv::Mat srcImage = gray.clone();
                    cv::Mat dstImage;
                    for (double sigma = 1; sigma < 6; sigma += 2) {
                        for (float weight = 0.5; weight < 0.8; weight += 0.1) {
                            GaussianBlur(srcImage, dstImage, cv::Size(0, 0), sigma);
                            dstImage = (srcImage - (1 - weight) * dstImage) / weight;
                            zbar::zbar_image_set_data(zbarimage, dstImage.data, cut_width * cut_height, NULL);
                            n = zbar_scan_image(scanner, zbarimage);
                            if (n > 0) {
                                fprintf(stdout, "ZBAR scan success use USM sigma=%f weight=%f\n", sigma, weight);
                                goto AFTER_SCAN;
                            } else {
                                dstImage.release();
                            }
                        }
                    }
                }

                //cut区域外扩
                if (n < 1) {
                    for (int pix = 15; pix < 35; pix++)
                    {
                        int x1_neww = (obj.rect.x - pix) > 0?(obj.rect.x - pix):0;
                        int x2_neww = (obj.rect.x + obj.rect.width + pix) < image_org.cols?(obj.rect.x + obj.rect.width + pix):image_org.cols;
                        int y1_neww = (obj.rect.y - pix) > 0?(obj.rect.y - pix):0;
                        int y2_neww = (obj.rect.y + obj.rect.height + pix) < image_org.rows?(obj.rect.y + obj.rect.height + pix):image_org.rows;
                        cv::Rect roi_new(x1_neww,y1_neww,x2_neww-x1_neww,y2_neww-y1_neww);
                        cv::Mat dstImage = image_org(roi_new);
                        int cut_width  = x2_neww-x1_neww;
                        int cut_height = y2_neww-y1_neww;
                        cv::Mat gray;
                        cv::cvtColor(dstImage, gray, cv::COLOR_BGR2GRAY);
                        zbar::zbar_image_set_size(zbarimage, cut_width, cut_height);
                        zbar::zbar_image_set_data(zbarimage, gray.data, cut_width * cut_height, NULL);
                        n = zbar_scan_image(scanner, zbarimage);
                        if (n > 0) {
                            fprintf(stdout, "ZBAR scan success use size of %dx%d\n", cut_width, cut_height);
                            goto AFTER_SCAN;
                        } else {
                            dstImage.release();
                        }
                    }

                }

                //cut区域放大
                if (n < 1) {
                    cv::Mat srcImage = gray.clone();
                    cv::Mat dstImage;
                    cv::resize(srcImage, dstImage, cv::Size(ZBAR_DECODE_W, ZBAR_DECODE_H));
                    zbar::zbar_image_set_size(zbarimage, ZBAR_DECODE_W, ZBAR_DECODE_H);
                    zbar::zbar_image_set_data(zbarimage, dstImage.data, ZBAR_DECODE_W * ZBAR_DECODE_H, NULL);
                    n = zbar_scan_image(scanner, zbarimage);
                    if (n > 0) {
                        fprintf(stdout, "ZBAR scan success use size of %dx%d\n", ZBAR_DECODE_W, ZBAR_DECODE_H);
                        goto AFTER_SCAN;
                    } else {
                        dstImage.release();
                    }
                }

                AFTER_SCAN:
                    const zbar::zbar_symbol_t * symbol;
                    symbol = zbar::zbar_image_first_symbol(zbarimage);
                    for (; symbol; symbol = zbar::zbar_symbol_next(symbol)) {
                        zbar::zbar_symbol_type_t typ = zbar_symbol_get_type(symbol);
                        const char *data = zbar_symbol_get_data(symbol);
                        int pointCount = zbar_symbol_get_loc_size(symbol);
                        fprintf(stdout, "Decode data:[%s], type:[%s]\n", const_cast<char *>(data),const_cast<char *>(zbar_get_symbol_name(typ)));
                    }
                if (n>0)
                {
                    success = true;
                }
            }
            if(success)
            {
                total_decode_count +=1;
            }
        fprintf(stdout, "--------------------------------------\n");
        }
        fprintf(stdout, "Total pics:%d\n", files_vector.size());
        fprintf(stdout, "Total decode count:%d\n", total_decode_count);
        fprintf(stdout, "Decode rate:%.1f%\n",total_decode_count*100.0/files_vector.size());

        zbar::zbar_image_destroy(zbarimage);
        zbar::zbar_image_scanner_destroy(scanner);
        middleware::free_io(&io_data);
        return AX_ENGINE_DestroyHandle(handle);
    }
} // namespace ax

int main(int argc, char* argv[])
{
    cmdline::parser cmd;
    cmd.add<std::string>("model", 'm', "joint file(a.k.a. joint model)", true, "");
    cmd.add<std::string>("image", 'i', "image path", true, "");
    cmd.add<std::string>("output", 'o', "out path", true, "./res");
    cmd.add<std::string>("size", 'g', "input_h, input_w", false, std::to_string(DEFAULT_IMG_H) + "," + std::to_string(DEFAULT_IMG_W));

    cmd.add<int>("repeat", 'r', "repeat count", false, DEFAULT_LOOP_COUNT);
    cmd.parse_check(argc, argv);

    // 0. get app args, can be removed from user's app
    auto model_file = cmd.get<std::string>("model");
    auto image_dir = cmd.get<std::string>("image");
    auto output_dir = cmd.get<std::string>("output");

    auto model_file_flag = utilities::file_exist(model_file);
    auto image_path_flag = utilities::path_exist(image_dir);
    if (!model_file_flag || !image_path_flag)
    {
        auto show_error = [](const std::string& kind, const std::string& value) {
            fprintf(stderr, "Input file %s(%s) is not exist, please check it.\n", kind.c_str(), value.c_str());
        };

        if (!model_file_flag) { show_error("model", model_file); }
        if (!image_path_flag) { show_error("image", image_dir); }

        return -1;
    }

    if (!utilities::path_exist(output_dir))
    {
        utilities::create_dir(output_dir);
    }

    auto input_size_string = cmd.get<std::string>("size");

    std::array<int, 2> input_size = {DEFAULT_IMG_H, DEFAULT_IMG_W};

    auto input_size_flag = utilities::parse_string(input_size_string, input_size);

    if (!input_size_flag)
    {
        auto show_error = [](const std::string& kind, const std::string& value) {
            fprintf(stderr, "Input %s(%s) is not allowed, please check it.\n", kind.c_str(), value.c_str());
        };

        show_error("size", input_size_string);

        return -1;
    }

    auto repeat = cmd.get<int>("repeat");

    // 1. print args
    fprintf(stdout, "--------------------------------------\n");
    fprintf(stdout, "model file : %s\n", model_file.c_str());
    fprintf(stdout, "image dir : %s\n", image_dir.c_str());
    fprintf(stdout, "img_h, img_w : %d %d\n", input_size[0], input_size[1]);
    fprintf(stdout, "--------------------------------------\n");

    // 3. sys_init
    AX_SYS_Init();

    // 4. -  engine model  -  can only use AX_ENGINE** inside
    {
        // AX_ENGINE_NPUReset(); // todo ??
        ax::run_model(model_file, image_dir, repeat, input_size[0], input_size[1], output_dir);

        // 4.3 engine de init
        AX_ENGINE_Deinit();
        // AX_ENGINE_NPUReset();
    }
    // 4. -  engine model  -

    AX_SYS_Deinit();
    return 0;
}
