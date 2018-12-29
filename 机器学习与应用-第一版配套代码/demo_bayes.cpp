#include "opencv2/opencv.hpp"
#include "opencv2/contrib/contrib.hpp"


using namespace cv;

int main(int argc, char **argv)
{
	const int kWidth = 512; // 分类结果图像的宽度
	const int kHeight = 512; // 分类结果图像的高度
	Vec3b red(0, 0, 255), green(0, 255, 0), blue(255, 0, 0); // 显示分类结果的3种颜色
	// 用于显示分类结果的图像
	Mat image = Mat::zeros(kHeight, kWidth, CV_8UC3);  
	// 为训练样本标签赋值
	int labels[30];
	for (int i  = 0 ; i < 10; i++)
		labels[i] = 1; // 前面10个样本为第1类
	for (int i = 10; i < 20; i++)
		labels[i] = 2; // 中间10个样本为第2类
	for (int i = 20; i < 30; i++)
		labels[i] = 3; // 最后10个样本为第3类
	Mat trainResponse(30, 1, CV_32SC1, labels);
	// 生成训练样本特征向量数组
	float trainDataArray[30][2];
	RNG rng; // 用于生成随机数
	for (int i = 0; i < 10; i++)
	{
// 首先生成第1类样本的特征向量
// x和y都服从正态分布，用随机数生成样本的特征值
// gaussian函数生成指定标准差、均值为0的正态分布数，这里标准差为30
		trainDataArray[i][0] = 250 + static_cast<float>(rng.gaussian(30));
		trainDataArray[i][1] = 250 + static_cast<float>(rng.gaussian(30));
	}
	for (int i = 10; i < 20; i++)
	{
// 生成第2类样本的特征向量
		trainDataArray[i][0] = 150 + static_cast<float>(rng.gaussian(30));
		trainDataArray[i][1] = 150 + static_cast<float>(rng.gaussian(30));
	}
	for (int i = 20; i < 30; i++)
	{
// 生成第3类样本的特征向量
		trainDataArray[i][0] = 320 + static_cast<float>(rng.gaussian(30));
		trainDataArray[i][1] = 150 + static_cast<float>(rng.gaussian(30));
	}
	Mat trainData(30, 2, CV_32FC1, trainDataArray);
	CvNormalBayesClassifier bayesClassifier;
	// 训练贝叶斯分类器
	bayesClassifier.train(trainData, trainResponse);
	// 对图像内所有点(i, j)即特征向量(x, y)进行预测，在这里i是y，j是x
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
// 生成测试样本特征向量
			Mat sampleMat = (Mat_<float>(1, 2) << j, i); 	
// 用贝叶斯分类器进行预测
			float response = bayesClassifier.predict(sampleMat); 
// 根据预测结果显示不同的颜色
			if (response == 1)
				image.at<Vec3b>(i, j) = red;
			else if (response == 2)
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
		else if (labels[i] == 2)
			circle(image, pt, 5, Scalar::all(128), -1, 8);
		else
			circle(image, pt, 5, Scalar::all(255), -1, 8);
	}
// 显示分类结果图像，水平方向为x，垂直方向为y
	imshow("Bayessian classifier demo", image);
	waitKey(0);
	return 0;
}