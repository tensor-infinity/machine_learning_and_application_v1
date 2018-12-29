#include "opencv2/opencv.hpp"
#include "opencv2/contrib/contrib.hpp"


using namespace cv;

int main(int argc, char **argv)
{
	const int kClassNum = 2; // 类型数
	const int kWidth = 512, kHeight = 512;
	Vec3b red(0, 0, 255), green(0, 255, 0), blue(255, 0, 0);
	Mat image = Mat::zeros(kHeight, kWidth, CV_8UC3);  
	// 训练样本标签数组
	int labels[150];
	for (int i  = 0 ; i < 75; i++)
		labels[i] = 0;
	for (int i = 75; i < 150; i++)
		labels[i] = 1;
	std::vector<int> trainResponse;   
	for (int i = 0; i < 150; i++)
		trainResponse.push_back(labels[i]);
	// 训练样本特征向量数组
	double trainDataArray[150][2];
	RNG rng;
	for (int i = 0; i < 75; i++)
	{
		trainDataArray[i][0] = 350 +rng.gaussian(30);
		trainDataArray[i][1] = 350 + rng.gaussian(30);
	}
	for (int i = 75; i < 150; i++)
	{
		trainDataArray[i][0] = 150 + rng.gaussian(30);
		trainDataArray[i][1] = 150 + rng.gaussian(30);
	}
	Mat trainData(150, 2, CV_64FC1, trainDataArray);
	// 计算LDA投影，投影后为一维的，即kClassNum - 1
	LDA lda(trainData, trainResponse, kClassNum - 1);
	Mat eigenVector = lda.eigenvectors().clone(); // 获取特征向量
	vector<Mat> classMean(kClassNum);  
	vector<int> classCount(kClassNum); 
// 下面的代码用来计算量每个类的均值向量投影后的值
	for (int i = 0; i < kClassNum; i++)  
	{  
		classMean[i] = Mat::zeros(1, trainData.cols, CV_64FC1);  //初始化类中均值为0  
		classCount[i] = 0;  //每一类中的样本数
	}  
	Mat sample;  
	for (int i = 0;i < trainData.rows; i++)  
	{	// 先计算每类样本特征向量的累加值  
		sample = trainData.row(i);  
		if(labels[i]==0)    
		{     
			add(classMean[0], sample, classMean[0]);  
			classCount[0]++;  
		}  
		else   
		{  
			add(classMean[1], sample, classMean[1]);  
			classCount[1]++;  
		}  
	}  
// 然后除以每类样本的数量，得到均值向量
	for (int i = 0; i < kClassNum; i++)   
		classMean[i].convertTo(classMean[i], CV_64FC1, 
1.0/static_cast<float>(classCount[i]));  
	// 两个类投影后的中心
	vector<Mat> cluster(kClassNum);
	// 计算两个类投影后的中心，在这里是一维的
	// 类均值和投影矩阵相乘，得到投影后的类中心，在这里是一维的点
	for (int i = 0; i < kClassNum; i++)  
		cluster[i] = classMean[i]*eigenVector;  
	// 对图像内所有点进行预测
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			Mat sampleMat = (Mat_<double>(1, 2) << j, i); 
			Mat projection = Mat::zeros(1,1,CV_64FC1);
// 先计算样本向量的投影
			projection = sampleMat*eigenVector; 
			double temp = projection.ptr<double>(0)[0];
			// 然后比较与哪个类的投影中心更接近，确定分类结果
			int response = (fabs(temp - cluster[0].ptr<double>(0)[0]) < fabs(temp - 
cluster[1].ptr<double>(0)[0])) ? 0 : 1; 
			if (response == 0)
				image.at<Vec3b>(i, j) = green;
			else 
				image.at<Vec3b>(i, j) = blue;
		}
	}
// 显示两类训练样本
	for (int i = 0; i < trainData.rows; i++)
	{
		const double* v = trainData.ptr<double>(i);
		Point pt = Point((int)v[0], (int)v[1]);

		if (labels[i] == 0)
			circle(image, pt, 5, Scalar::all(0), -1, 8); 
		else
			circle(image, pt, 5, Scalar::all(255), -1, 8);
	}
	imshow("LDA classifier demo", image);
	waitKey(0);
	return 0;
}