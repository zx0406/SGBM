
#include <iostream>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <opencv2/opencv.hpp>  
#include <iostream>  
#include <stdio.h>
#include <math.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;
int main()
{

	IplImage* img1 = cvLoadImage("im21.png", 0);
	IplImage* img2 = cvLoadImage("im61.png", 0);

	Ptr<StereoSGBM>sgbm = StereoSGBM::create(0, 16, 3);
	int SADWindowSize = 9;

	sgbm->setPreFilterCap(15);

	sgbm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 3);
	int cn = img1->nChannels;
	int numberOfDisparities = 64;
	sgbm->setP1(8 * cn * SADWindowSize * SADWindowSize);
	sgbm->setP2(32 * cn * SADWindowSize * SADWindowSize);
	sgbm->setMinDisparity(0);
	sgbm->setNumDisparities(numberOfDisparities);
	sgbm->setUniquenessRatio(10);
	sgbm->setSpeckleRange(100);
	sgbm->setSpeckleRange(32); 
	sgbm->setDisp12MaxDiff(1);

	Mat disp, disp8;
	int64 t = getTickCount();



	sgbm->compute(cvarrToMat(img1),cvarrToMat(img2), disp);//IplImage 转化问题 cvarrToMat函数使用
	t = getTickCount() - t;
	cout << "Time elapsed:" << t * 1000 / getTickFrequency() << endl;
	disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities * 16.));

	namedWindow("left", 1);
	
	cvShowImage("left", img1);
	namedWindow("right", 1);
	cvShowImage("right", img2);
	namedWindow("disparity", 1);
	imshow("disparity", disp8);
	waitKey();
	imwrite("sgbm_disparity.png", disp8);
	cvDestroyAllWindows();
	return 0;
}
