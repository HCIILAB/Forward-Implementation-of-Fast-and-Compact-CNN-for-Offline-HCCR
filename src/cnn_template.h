/*
 * Copyright © DLVC 2017.
 *
 * Paper: Building Fast and Compact Convolutional Neural Networks for Offline Handwritten Chinese Character Recognition (arXiv:1702.07975 [cs.CV])
 * Authors: Xuefeng Xiao, Lianwen Jin, Yafeng Yang, Weixin Yang, Jun Sun, Tianhai Chang
 * Email: xiaoxuefengchina@gmail.com
 */
#ifndef CNN_TEMPLATE_H
#define CNN_TEMPLATE_H
#include <mkl.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <map>
#include <vector>
#include "save_bit.hpp"

using namespace std;

struct convParams{
	int inputChannels;
	int inputH;
	int inputW;
	int outputChannels;
	int outputH;
	int outputW;
	int kernelH;
	int kernelW;
	int kernelStrideH;
	int kernelStrideW;
	int padH;
	int padW;
};  
struct dataParams{
    float scale_factor;
    int channels;
    int h;
    int w;
};
struct batchNormParams{
    char bottom_layer_name[50];
    char top_layer_name[50];
    int channels;
};
struct PReLUParams{
    char bottom_layer_name[50];
    char top_layer_name[50];
	int channels;
};
struct fcParams{
	int inputSize;
	int outputSize;
	int fcNeuralUnit;
	int length;
};
struct sparseFcParams{
	int inputSize;
	int outputSize;
	int fcNeuralUnit;
	int length;
};
struct poolParams{
	int channels;
	int inputH;
	int inputW;
	int outputH;
	int outputW;
	int kernel;
	int kernelStride;
	int padH;
	int padW;
};

class ParamsBin{
public:
	convParams cParams;
	dataParams dParams;
	fcParams fParams;
	poolParams pParams;
	batchNormParams bnParams;
	PReLUParams preluParams;
};

const CBLAS_ORDER Order = CblasRowMajor;
const CBLAS_TRANSPOSE TransA = CblasNoTrans;
const CBLAS_TRANSPOSE TransB = CblasNoTrans;

const float alpha = 1;
const float beta = 0;

const int get_log_two(const int data);


void CHECK_IF(const bool a);
void CHECK_INFO(const bool a, const std::string info);

int count_non_zero_num(float* weights, vector<int>& non_zero_idx, int weightSize);
void CSR_coding(float *A, float *a, int *ia, int *ja, vector<int> &non_zero_idx, const int M, const int N);
bool unpack_cluster_blob(float* weights, FILE* &fp, int weightSize);
bool unpack_cluster_blob_fc(float* weights,FILE* &fp,float* &weight_sparse,float* &mask,float* &array_each_output,fcParams &fParams_);
void reconvery_blob_data(float* weights,const float* cluster_center,const int* array_cluster_label_index,const int diff_index_pair_length,\
	const int weights_num,const byte* array_diff);

class CNN{
private:
	inline bool is_a_ge_zero_and_a_lt_b(int a, int b);
	inline void im2col(const float* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		float* data_col);
	
	void meanSubtractAndScale(const unsigned char * image,const int input_image_size , string tName);
	void conv_BLAS_template(const string bName, const string tName);
	void conv_BLAS_template_BN_PReLU(const string bName, const string tName,const string bnName,const string preluName);
	void max_pooling_extend_loop_k3(const string bName, const string tName);
	void fc_template_sparse_mv(const string bName, const string tName, bool isRelu);
	void fc_template_BN_PReLU_sparse_mv(const string bName, const string tName,const string bnName,const string preluName);

	inline int* judge(const float *input, int len);
	void LoadParams(const string paramsFileAddr);
	float ReLU(float data);

public:
	CNN();
	CNN(const string addr);
	~CNN();
	int* forward(unsigned char* data,const int dataSize);

private:
	map<string, int> layernames_;
	vector<string> layertype_;
	vector<ParamsBin> paramsBin_;		
	vector<int> name2idx_;
	vector<vector<float*> > params_;
	vector<float*> top_;			
	
	pair<int, float> *result_pair_array;
	int* result_array;

};

#endif
