#include "opencv2/opencv.hpp"
#include "opencv2/contrib/contrib.hpp"


using namespace cv;

int main(int argc, char **argv)
{
	const int kWidth = 512, kHeight = 512; // 显示分类结果的图像的高度和宽度
	Vec3b red(0, 0, 255), green(0, 255, 0), blue(255, 0, 0); // 显示分类结果的颜色
	Mat image = Mat::zeros(kHeight, kWidth, CV_8UC3);  
	// 为训练样本标签赋值，每类样本50个
	int labels[150];
	for (int i  = 0 ; i < 50; i++)
		labels[i] = 1;
	for (int i = 50; i < 100; i++)
		labels[i] = 2;
	for (int i = 100; i < 150; i++)
		labels[i] = 3;
	Mat trainResponse(150, 1, CV_32SC1, labels);
	// 为训练样本特征向量数组赋值
	float trainDataArray[150][2];
	RNG rng;
	for (int i = 0; i < 50; i++)
	{
		trainDataArray[i][0] = 250 + static_cast<float>(rng.gaussian(30));
		trainDataArray[i][1] = 250 + static_cast<float>(rng.gaussian(30));
	}
	for (int i = 50; i < 100; i++)
	{
		trainDataArray[i][0] = 150 + static_cast<float>(rng.gaussian(30));
		trainDataArray[i][1] = 150 + static_cast<float>(rng.gaussian(30));
	}
	for (int i = 100; i < 150; i++)
	{
		trainDataArray[i][0] = 320 + static_cast<float>(rng.gaussian(30));
		trainDataArray[i][1] = 150 + static_cast<float>(rng.gaussian(30));
	}
	Mat trainData(150, 2, CV_32FC1, trainDataArray);
// 随机森林的训练参数，在源代码分析中会详细讲解。在这里，决策树的数量为5
	CvRTParams params = CvRTParams(5, 2, 0, false, 2, NULL, true, 0, 5, 0, 
		CV_TERMCRIT_ITER); 
	CvRTrees rtrees;    
	// 训练随机森林	
	rtrees.train(trainData, CV_ROW_SAMPLE, trainResponse,  cv::Mat(), 
		cv::Mat(), cv::Mat(), cv::Mat(), params);   
	// 对图像内所有点进行预测，并显示不同的颜色
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << j, i); 
			float response = rtrees.predict(sampleMat); 
			if (response == 1)
				image.at<Vec3b>(i, j) = red;
			else if (response == 2)
				image.at<Vec3b>(i, j) = green;
			else
				image.at<Vec3b>(i, j) = blue;
		}
	}
// 显示3类训练样本
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

	imshow("random forest classifier demo", image);
	waitKey(0);
	return 0;
}