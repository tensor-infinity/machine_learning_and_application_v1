#include "opencv2/opencv.hpp"
#include "opencv2/contrib/contrib.hpp"


using namespace cv;

int main(int argc, char **argv)
{
	const int kWidth = 512, kHeight = 512;
	Vec3b red(0, 0, 255), green(0, 255, 0), blue(255, 0, 0);
	Mat image = Mat::zeros(kHeight, kWidth, CV_8UC3); 
	// 训练样本标签数组
	int labels[150];
	for (int i  = 0 ; i < 75; i++)
		labels[i] = 1;
	for (int i = 75; i < 150; i++)
		labels[i] = 2;
	Mat trainResponse(150, 1, CV_32SC1, labels);
	// 训练样本特征向量数组
	float trainDataArray[150][2];
	RNG rng;
	for (int i = 0; i < 75; i++)
	{
		trainDataArray[i][0] = 250 + static_cast<float>(rng.gaussian(30));
		trainDataArray[i][1] = 250 + static_cast<float>(rng.gaussian(30));
	}
	for (int i = 75; i < 150; i++)
	{
		trainDataArray[i][0] = 150 + static_cast<float>(rng.gaussian(30));
		trainDataArray[i][1] = 150 + static_cast<float>(rng.gaussian(30));
	}
	Mat trainData(150, 2, CV_32FC1, trainDataArray);
	float priors[2] = {1, 1};     
// AdaBoost训练参数，源代码分析中会解释，在这里我们设置弱分类器的个数
// 为10
	CvBoostParams params( CvBoost::REAL, 10, 0.95,  5,  false, priors);     
	CvBoost boost;    
// 训练分类器
	boost.train(trainData, CV_ROW_SAMPLE, trainResponse,  cv::Mat(), cv::Mat(), 
		cv::Mat(), cv::Mat(), params);   
	// 对图像内所有点进行预测，显示不同的颜色
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << j, i); 
// 调用AdaBoost的预测函数
			float response = boost.predict(sampleMat); 
// 根据预测结果显示不同颜色
			if (response == 1)
				image.at<Vec3b>(i, j) = green;
			else 
				image.at<Vec3b>(i, j) = blue;
		}
	}
// 显示训练样本
	for (int i = 0; i < trainData.rows; i++)
	{
		const float* v = trainData.ptr<float>(i);
		Point pt = Point((int)v[0], (int)v[1]);
		if (labels[i] == 1)
			circle(image, pt, 5, Scalar::all(0), -1, 8); 
		else
			circle(image, pt, 5, Scalar::all(255), -1, 8);
	}
	imshow("AdaBoost classifier demo", image);
	waitKey(0);
	return 0;
}