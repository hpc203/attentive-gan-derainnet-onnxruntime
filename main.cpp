#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;


class Derainnet
{
public:
	Derainnet(string modelpath);
	Mat detect(const Mat& frame);
private:
	vector<float> input_image;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "Attentive Generative Adversarial Network for Raindrop Removal from A Single Image");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	const vector<const char*> input_names = {"input"};
	const vector<const char*> output_names = {"output"};
    int inpWidth;
	int inpHeight;
    void preprocess(const Mat& frame);
};

Derainnet::Derainnet(string model_path)
{
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    // std::wstring widestr = std::wstring(model_path.begin(), model_path.end());   ////windows写法
	// ort_session = new Session(env, widestr.c_str(), sessionOptions);           ////windows写法
    ort_session = new Session(env, model_path.c_str(), sessionOptions);          ////linux写法

    size_t numInputNodes = ort_session->GetInputCount();
    AllocatorWithDefaultOptions allocator;
    vector<vector<int64_t>> input_node_dims;
	for (int i = 0; i < numInputNodes; i++)
	{
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}

    this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
}

void Derainnet::preprocess(const Mat& frame)
{
    Mat dstimg;
    resize(frame, dstimg, Size(this->inpWidth, this->inpHeight));
    dstimg.convertTo(dstimg, CV_32FC3, 1 / 127.5f, -1.0f);

	vector<Mat> rgbChannels(3);
    split(dstimg, rgbChannels);
	const int image_area = dstimg.rows * dstimg.cols;
    this->input_image.clear();
	this->input_image.resize(1 * 3 * image_area);
    int single_chn_size = image_area * sizeof(float);
	memcpy(this->input_image.data(), (float *)rgbChannels[0].data, single_chn_size);
    memcpy(this->input_image.data() + image_area, (float *)rgbChannels[1].data, single_chn_size);
    memcpy(this->input_image.data() + image_area * 2, (float *)rgbChannels[2].data, single_chn_size);
}

Mat Derainnet::detect(const Mat& frame)
{
	this->preprocess(frame);

	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };
    auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, this->input_image.data(), this->input_image.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	
    std::vector<int64_t> out_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();  ////输出形状是1,h,w,3
    const int out_h = out_shape[1];
    const int out_w = out_shape[2];
	float* pred = ort_outputs[0].GetTensorMutableData<float>();
    vector<int> newshape = {out_h, out_w, 3};           
    Mat out = Mat(newshape, CV_32FC1, pred);
    Mat chw_mat;
    cv::transposeND(out, {2, 0, 1}, chw_mat);
	const int channel_step = out_h * out_w;
    float* pdata = (float*)chw_mat.data;
	Mat bmat(out_h, out_w, CV_32FC1, pdata);
	Mat gmat(out_h, out_w, CV_32FC1, pdata + channel_step);
	Mat rmat(out_h, out_w, CV_32FC1, pdata + 2 * channel_step);

    double minVal, maxVal;    // 最大值，最小值
	cv::Point minIdx, maxIdx;    // 最小值坐标，最大值坐标     
	cv::minMaxLoc(bmat, &minVal, &maxVal, &minIdx, &maxIdx);
    bmat = (bmat - minVal)*255.f / (maxVal - minVal);

	cv::minMaxLoc(gmat, &minVal, &maxVal, &minIdx, &maxIdx);
    gmat = (gmat - minVal)*255.f / (maxVal - minVal);

    cv::minMaxLoc(rmat, &minVal, &maxVal, &minIdx, &maxIdx);
    rmat = (rmat - minVal)*255.f / (maxVal - minVal);

    vector<Mat> channel_mats = {bmat, gmat, rmat};
    Mat dstimg;
	merge(channel_mats, dstimg);
	dstimg.convertTo(dstimg, CV_8UC3);
	resize(dstimg, dstimg, Size(frame.cols, frame.rows));
    return dstimg;
}


int main()
{
	Derainnet mynet("weights/attentive_gan_derainnet_240x360.onnx");  
	string imgpath = "sample.png";  ///文件路径写正确，程序才能正常运行的
	Mat srcimg = imread(imgpath);

	Mat dstimg = mynet.detect(srcimg);
	
	imwrite("result.jpg", dstimg);

    // namedWindow("srcimg", WINDOW_NORMAL);
	// imshow("srcimg", srcimg);
	// static const string kWinName = "Deep learning use onnxruntime";
	// namedWindow(kWinName, WINDOW_NORMAL);
	// imshow(kWinName, dstimg);
	// waitKey(0);
	// destroyAllWindows();
}