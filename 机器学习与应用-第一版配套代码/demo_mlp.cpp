#include "opencv2/opencv.hpp"
#include "opencv2/contrib/contrib.hpp"


using namespace cv;

int main(int argc, char **argv)
{
	const int kWidth = 512, kHeight = 512;
	Vec3b red(0, 0, 255), green(0, 255, 0), blue(255, 0, 0);
	Mat image = Mat::zeros(kHeight, kWidth, CV_8UC3);  
	// 为3类训练样本的标签赋值
	float labels[150][3];
	for (int i  = 0 ; i < 50; i++)
	{
		labels[i][0] =  1.0f;
		labels[i][1] = -1.0f;
		labels[i][2] = -1.0f;
	}
	for (int i = 50; i < 100; i++)
	{
		labels[i][0] = -1.0f;
		labels[i][1] =  1.0f;
		labels[i][2] = -1.0f;
	}
	for (int i = 100; i < 150; i++)
	{
		labels[i][0] = -1.0f;
		labels[i][1] = -1.0f;
		labels[i][2] =  1.0f;
	}
	Mat trainResponse(150, 3, CV_32FC1, labels);
	// 生成训练样本特征向量数组
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
	CvANN_MLP mlp;
// 神经网络有3层，第一层2个神经元，对应于二维的特征向量。第二层有
// 6个神经元，第三层有3个神经元，对应分类问题中的3个类别
	Mat layerSizes=(Mat_<int>(1,3) << 2, 6, 3);
	CvANN_MLP_TrainParams params;
// 神经网络的训练参数，在后面会详细解释
	params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 
		1000, 0.001 );
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;  
	params.bp_dw_scale = 0.1;  
	params.bp_moment_scale = 0.1;  
// 创建神经网络
	mlp.create(layerSizes, CvANN_MLP::SIGMOID_SYM);
	// 训练神经网络
	mlp.train(trainData, trainResponse,  Mat(), Mat(), params);
	// 对图像内所有点(i,j )进行预测，根据分类结果显示不同的颜色
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << j, i); 	
			Mat predictResult(1, 3, CV_32FC1);
// 用神经网络预测，predictResult是预测输出向量
			mlp.predict(sampleMat, predictResult); 
			Point maxLoc;
			double maxVal;
// 训练向量的最大分量，maxVal是返回的最大分量值，maxLoc是最大值
// 对应的分量号，即分类结果
			minMaxLoc(predictResult, 0, &maxVal, 0, &maxLoc);
// 根据分类结果显示不同的颜色
			if (maxLoc.x == 0)
				image.at<Vec3b>(i, j) = red;
			else if (maxLoc.x == 1)
				image.at<Vec3b>(i, j) = blue;
			else
				image.at<Vec3b>(i, j) = green;
		}
	}
// 显示训练样本
	for (int i = 0; i < trainData.rows; i++)
	{
		const float* v = trainData.ptr<float>(i);
		Point pt = Point((int)v[0], (int)v[1]);
		if (labels[i][0] == 1)
			circle(image, pt, 5, Scalar::all(0), -1, 8); 
		else if (labels[i][1] == 1)
			circle(image, pt, 5, Scalar::all(128), -1, 8);
		else
			circle(image, pt, 5, Scalar::all(255), -1, 8);
	}
	imshow("MLP classifier demo", image);
	waitKey(0);
	return 0;
}