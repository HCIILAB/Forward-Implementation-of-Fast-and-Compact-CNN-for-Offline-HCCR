/*
 * Copyright © DLVC 2017.
 *
 * Paper: Building Fast and Compact Convolutional Neural Networks for Offline Handwritten Chinese Character Recognition (arXiv:1702.07975 [cs.CV])
 * Authors: Xuefeng Xiao, Lianwen Jin, Yafeng Yang, Weixin Yang, Jun Sun, Tianhai Chang
 * Email: xiaoxuefengchina@gmail.com
 */
#include "cnn_template.h"
#include <string.h>
#include <time.h>
#include <fstream>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <map>
#include <stdio.h>  
#include <string.h> 
#include <stdlib.h>  
#include <dirent.h>  
#include <sys/stat.h>  
#include <unistd.h>  
#include <sys/types.h> 

using namespace std;
using namespace cv;

#define SHOWTIME

int sum_weight_size = 0;
int sum_zero_num = 0;
int sum_compress_byte = 0;

const int CharacterWidth = 96;
const int CharacterHeight = 96;
const int CharacterChannel = 1;

int main()
{
	//the save path of compress binary file that store the params of our network
	const string paramsAddr = "./src/params_bin/params_compress.bin";

	int image_size = CharacterChannel * CharacterWidth * CharacterHeight;
	unsigned char buffer[image_size];

	CNN net(paramsAddr);	//build a forward convolutional neural network
	printf("WholeNetwork:\t\t");
	printf("params num = %d\t", sum_weight_size);								//output the weight size of whole network
	printf("pR = %.2f%%\t",(1 - sum_zero_num*1.0/sum_weight_size)* 100);		//output the size of the non-zero weight
	printf("p+c = %.2f%%\n", sum_compress_byte*1.0/4.0/sum_weight_size * 100);	//output the equivalent weight size of compress bin
	printf("Finish loading network parameters.\n\n");

	time_t start_time = 0;
	time_t end_time = 0;
	time_t sum_time = 0;
	int accuracy = 0;

	ifstream ImgFile("./ImageName_Label.txt", ios::in);  //ImageName_Label.txt save the name of the images and their corresponding label.
	if (!ImgFile.is_open())
	{
		cout << "Open ImgFile error" << endl;
		exit(1);
	}

	int ImgNum = 0;
	string ImgName_Label;
	string SaveImgPath_Format = "./Image/";
	printf("Start forward process...\n");
	while(getline(ImgFile, ImgName_Label)) 			//get ImgName from saved Test_Image_Name.txt
	{
		int idx = ImgName_Label.find(" ");
		string ImgName(ImgName_Label.substr(0, idx));		//extract image name
		int ImgLabel = atoi((ImgName_Label.substr(idx+1)).c_str());	//extract label of input image

		string SaveImgPath;
		SaveImgPath = SaveImgPath_Format + ImgName;
		IplImage* img = cvLoadImage(SaveImgPath.c_str(), CV_LOAD_IMAGE_ANYCOLOR);	//load test image from SaveImgPath
		if (img == NULL){
			cout << "The image does not exit!" << endl;
			continue;
		}

//load image to buffer
		int pos = 0;
	    uchar* data = (uchar *) img->imageData;
	    int step = img->widthStep / sizeof(uchar);
	    for (int m = 0; m < img->height; m++) {
	        for (int n = 0; n < img->width; n++) {
	            buffer[pos++]=data[m*step+n];
	        }
	    }

//start forward process and count the process time
		start_time = clock();
		int* result = net.forward(buffer,96*96);
		end_time = clock();
		sum_time += end_time - start_time;
		cvReleaseImage(&img);

//count the accuracy
		if (result[0] == ImgLabel){
			accuracy += 1;
		}

		ImgNum++;
		if (ImgNum%100 == 0){
			cout << ImgNum << " samples have been tested"<<endl;
		}
	}
	ImgFile.close();

	cout << "\nTesting each character sample spends " << sum_time / (ImgNum*1000.0) << " ms" << endl;
	cout << "Total tested characters num: " << ImgNum << "\ttop1 num: "<< accuracy <<endl;
	cout << "The final testing top1 accuracy is: " << accuracy*1.0 / ImgNum << endl;

}
