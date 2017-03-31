/*
 * Copyright © DLVC 2017.
 *
 * Paper: Building Fast and Compact Convolutional Neural Networks for Offline Handwritten Chinese Character Recognition (arXiv:1702.07975 [cs.CV])
 * Authors: Xuefeng Xiao, Lianwen Jin, Yafeng Yang, Weixin Yang, Jun Sun, Tianhai Chang
 * Email: xiaoxuefengchina@gmail.com
 */
#include "cnn_template.h"
#include <fstream>
#include <string.h>

#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))

void CHECK_IF(const bool a) {
	if(!a) { 
		printf( "Open File Error!\n"); 
		assert(0);
	}
}
void CHECK_INFO(const bool a, const std::string info) {
	if(!a) { 
		printf( "%s\n", info.c_str()); 
		assert(0);
	}
}
const int get_log_two(const int data){
	if(data==0)
		printf("cluster_center is error\n");
	int count=0,count_data=1;
	while(data>count_data){
		count_data*=2;
		count++;
	}
	return count;
}

CNN::CNN(){}

CNN::CNN(const string addr){
	LoadParams(addr);
	result_pair_array  = new pair<int, float>[3755]();
	result_array = new int[3755]();
}

CNN::~CNN(){
	int size = top_.size();
	for(int i = 0; i < size; ++i){
		delete[]top_[i];
	}
	size = params_.size();
	for(int i = 0; i < size; ++i){
		for(int j = 0; j < params_[i].size(); ++j){
			delete[]params_[i][j];
		}
	}
	delete[]result_array;
	delete[]result_pair_array;
}


inline bool CNN::is_a_ge_zero_and_a_lt_b(int a, int b) {
	return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

inline void CNN::im2col(const float* data_im, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	float* data_col) {

	const int output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
	const int output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
	const int channel_size = height * width;
	for (int channel = channels; channel--; data_im += channel_size) {
		for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
			for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
				int input_row = -pad_h + kernel_row ;
				for (int output_rows = output_h; output_rows; output_rows--) {
					if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
						for (int output_cols = output_w; output_cols; output_cols--) {
							*(data_col++) = 0;
						}
					}
					else {
						int input_col = -pad_w + kernel_col ;
						for (int output_col = output_w; output_col; output_col--) {
							if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
								*(data_col++) = data_im[input_row * width + input_col];
							}
							else {
								*(data_col++) = 0;
							}
							input_col += stride_w;
						}
					}
					input_row += stride_h;
				}
			}
		}
	}
}

void CNN::conv_BLAS_template(const string bName, const string tName){
	float *bottom = top_[layernames_[bName]];

	int idx = layernames_[tName];
	convParams  &conv_params = paramsBin_[idx].cParams;
	float *top = top_[idx];
	int paramsIdx = name2idx_[idx];
	CHECK_INFO(paramsIdx >= 0, "error");
	float *weight = params_[paramsIdx][0];
	float *bias = params_[paramsIdx][1];

	const int M = conv_params.outputChannels;
	const int K = conv_params.kernelH*conv_params.kernelW*conv_params.inputChannels;
	const int N = conv_params.outputH*conv_params.outputW;
	const int lda = K;
	const int ldb = N;
	const int ldc = N;

	const int output_map_size = conv_params.outputH*conv_params.outputW;
	memset(top, 0, sizeof(float) * output_map_size * conv_params.outputChannels);
	
	//extend the bottom data of every convolution window and save it into the input array
	float * input = new float[K*N]();
 	im2col(bottom, conv_params.inputChannels,
		 conv_params.inputH, conv_params.inputW, conv_params.kernelH, conv_params.kernelW,
		 conv_params.padH, conv_params.padW,
		 conv_params.kernelStrideH, conv_params.kernelStrideW,
		 input);
 	/********************************************************************************************************************/
 	/*	void cblas_sgemm(const CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, */
	/*					const int M, const int N, const int K, const float alpha, const float *A, const int lda, 		*/
	/*					const float *B, const int ldb, const float beta, float *C, const int ldc);						*/
	/*																													*/
	/*	Order: Specifies row-major (C) or column-major (Fortran) data ordering.											*/
	/*	Trans​A: Specifies whether to transpose matrix A.																*/
	/*	Trans​B: Specifies whether to transpose matrix B.																*/
	/*	M: Number of rows in matrices A and C.																			*/
	/*	N: Number of columns in matrices B and C.																		*/
	/*	K: Number of columns in matrix A; number of rows in matrix B.													*/
	/*	lda: The size of the first dimention of matrix A; if you are passing a matrix A[m][n], the value should be m.	*/
	/*	ldb: The size of the first dimention of matrix B; if you are passing a matrix B[m][n], the value should be m.	*/
	/*	ldc: The size of the first dimention of matrix C; if you are passing a matrix C[m][n], the value should be m.	*/
	/********************************************************************************************************************/
  	cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, weight, lda, input, ldb, beta, top, ldc);
	
	for (int count_out_feature_map = 0; count_out_feature_map < conv_params.outputChannels; ++count_out_feature_map){
		for (int count_row = 0; count_row<conv_params.outputH; ++count_row){
			for (int count_column = 0; count_column < conv_params.outputW; ++count_column){
				top[count_out_feature_map*output_map_size + count_row*conv_params.outputW + count_column] += bias[count_out_feature_map];
			}
		}
	}
	delete[] input;
}

void CNN::conv_BLAS_template_BN_PReLU(const string bName, const string tName,const string bnName,const string preluName){

	float *bottom = top_[layernames_[bName]];
	
	int idx = layernames_[tName];
	convParams  &conv_params = paramsBin_[idx].cParams;
	float *top = top_[idx];
	int paramsIdx = name2idx_[idx];
	CHECK_INFO(paramsIdx >= 0, "error");
	float *weight = params_[paramsIdx][0];
	float *bias = params_[paramsIdx][1];

	//read bn params
	idx = layernames_[bnName];
	batchNormParams  &bn_params = paramsBin_[idx].bnParams;
	paramsIdx = name2idx_[idx];
	CHECK_INFO(paramsIdx >= 0, "error");
	float *mean_extend = params_[paramsIdx][0];
	float *var = params_[paramsIdx][1];

	//read prelu params
	idx = layernames_[preluName];
	PReLUParams  &prelu_params = paramsBin_[idx].preluParams;
	paramsIdx = name2idx_[idx];
	CHECK_INFO(paramsIdx >= 0, "error");
	float *prelu_weight = params_[paramsIdx][0];

	CHECK_INFO(!strcmp(bn_params.bottom_layer_name, tName.c_str()), "error");
	CHECK_INFO(!strcmp(bn_params.top_layer_name, prelu_params.bottom_layer_name), "error");
	CHECK_INFO(!strcmp(prelu_params.top_layer_name, tName.c_str()), "error");


	const int output_map_size = conv_params.outputH*conv_params.outputW;
	memcpy(top, mean_extend, sizeof(float) * output_map_size * conv_params.outputChannels);		//copy mean_extend to top
	
	const int M = conv_params.outputChannels;	
	const int K = conv_params.kernelH*conv_params.kernelW*conv_params.inputChannels;
	const int N = conv_params.outputH*conv_params.outputW;
	const int lda = K;
	const int ldb = N;
	const int ldc = N;

	float* input = new float[K*N]();
 	im2col(bottom, conv_params.inputChannels,
		 conv_params.inputH, conv_params.inputW, conv_params.kernelH, conv_params.kernelW,
		 conv_params.padH, conv_params.padW,
		 conv_params.kernelStrideH, conv_params.kernelStrideW,
		 input);
 	cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, weight, lda, input, ldb, 1, top, ldc);	//add mean_extend

	for (int count_out_feature_map = 0; count_out_feature_map < conv_params.outputChannels; ++count_out_feature_map){
		for (int count_row = 0; count_row<conv_params.outputH; ++count_row){
		int index_base =count_out_feature_map*output_map_size + count_row*conv_params.outputW;
			for (int count_column = 0; count_column < conv_params.outputW; ++count_column){
				top[index_base + count_column] *=var[count_out_feature_map];
				
				// prelu
				top[index_base+ count_column] = max(top[index_base+ count_column], 0) + 
				prelu_weight[count_out_feature_map] *min(top[index_base + count_column], 0);
			}
		}
	}

	delete[] input;
}

void CNN::max_pooling_extend_loop_k3(const string bName, const string tName){
	float *bottom = top_[layernames_[bName]];
	int idx = layernames_[tName];
	poolParams  &pool_params = paramsBin_[idx].pParams;
	float *top = top_[idx];

	const int input_h_add_pad = pool_params.inputH + 2 * pool_params.padH;
	const int input_w_add_pad = pool_params.inputW + 2 * pool_params.padW;
	const int input_map_add_pad = input_h_add_pad *input_w_add_pad;
	const int output_map_size = pool_params.outputH*pool_params.outputW;
	memset(top, 0, sizeof(float)*pool_params.channels*output_map_size);

	float * input = new float[input_map_add_pad*pool_params.channels]();
	for (int c = 0; c < pool_params.channels; c++){
		for (int i = pool_params.padH; i<(input_h_add_pad - pool_params.padH); i++){
			for (int j = pool_params.padW; j<(input_w_add_pad - pool_params.padW); j++){
				input[c*input_map_add_pad + i*input_w_add_pad + j] = bottom[c*pool_params.inputH* pool_params.inputW + (i - pool_params.padH)*pool_params.inputW + (j - pool_params.padW)];
			}
		}
	}

	for(int c=0; c<pool_params.channels; ++c)
	{
		int hstart = 0;
		for(int ph=0; ph<pool_params.outputH; ++ph, hstart += pool_params.kernelStride)
		{
			int wstart = 0;
			for(int pw=0; pw<pool_params.outputW; ++pw, wstart += pool_params.kernelStride)
			{
				// not take account of padding situation
				int w_flag=0, h_flag=0;
				int hend=0;
				int wend=0;
				if(hstart+pool_params.kernel>input_w_add_pad) 
				{
					hend=input_w_add_pad;
					h_flag=1;
				}
				else
				{
					hend=hstart+pool_params.kernel;
					h_flag=0;
				}
				if(wstart+pool_params.kernel>input_w_add_pad)
				{
					wend=input_w_add_pad;
					w_flag=1;
				}
				else
				{
					wend=wstart+pool_params.kernel;
					w_flag=0;
				}
				int offsetOfFeatMap = c * input_map_add_pad;
				const int pool_index=c*output_map_size+ph*pool_params.outputW+pw;
				//loop unrolling for speeding up only for poolKernelSize=3
				if(w_flag==0 && h_flag==0)
				{							
					float maxIdx=input[offsetOfFeatMap+hstart*input_w_add_pad+wstart];
					maxIdx=max(input[offsetOfFeatMap+hstart*input_w_add_pad+wstart+1],maxIdx);
					maxIdx=max(input[offsetOfFeatMap+hstart*input_w_add_pad+wstart+2],maxIdx);
					maxIdx=max(input[offsetOfFeatMap+(hstart+1)*input_w_add_pad+wstart],maxIdx);
					maxIdx=max(input[offsetOfFeatMap+(hstart+1)*input_w_add_pad+wstart+1],maxIdx);
					maxIdx=max(input[offsetOfFeatMap+(hstart+1)*input_w_add_pad+wstart+2],maxIdx);
					maxIdx=max(input[offsetOfFeatMap+(hstart+2)*input_w_add_pad+wstart],maxIdx);
					maxIdx=max(input[offsetOfFeatMap+(hstart+2)*input_w_add_pad+wstart+1],maxIdx);
					maxIdx=max(input[offsetOfFeatMap+(hstart+2)*input_w_add_pad+wstart+2],maxIdx);
					top[pool_index]=maxIdx;

				}
				if(w_flag==1 && h_flag==0)
				{
					float maxIdx=input[offsetOfFeatMap+hstart*input_w_add_pad+wstart];
					maxIdx=max(input[offsetOfFeatMap+hstart*input_w_add_pad+wstart+1],maxIdx);
					maxIdx=max(input[offsetOfFeatMap+(hstart+1)*input_w_add_pad+wstart],maxIdx);
					maxIdx=max(input[offsetOfFeatMap+(hstart+1)*input_w_add_pad+wstart+1],maxIdx);
					maxIdx=max(input[offsetOfFeatMap+(hstart+2)*input_w_add_pad+wstart],maxIdx);
					maxIdx=max(input[offsetOfFeatMap+(hstart+2)*input_w_add_pad+wstart+1],maxIdx);
					top[pool_index]=maxIdx;
				}
				if(h_flag==1 && w_flag==0)
				{
					float maxIdx=input[offsetOfFeatMap+hstart*input_w_add_pad+wstart];
					maxIdx=max(input[offsetOfFeatMap+hstart*input_w_add_pad+wstart+1],maxIdx);
					maxIdx=max(input[offsetOfFeatMap+hstart*input_w_add_pad+wstart+2],maxIdx);
					maxIdx=max(input[offsetOfFeatMap+(hstart+1)*input_w_add_pad+wstart],maxIdx);
					maxIdx=max(input[offsetOfFeatMap+(hstart+1)*input_w_add_pad+wstart+1],maxIdx);
					maxIdx=max(input[offsetOfFeatMap+(hstart+1)*input_w_add_pad+wstart+2],maxIdx);
					top[pool_index]=maxIdx;
				}
				if(w_flag==1 && h_flag==1)
				{
					float maxIdx=input[offsetOfFeatMap+hstart*input_w_add_pad+wstart];
					maxIdx=max(input[offsetOfFeatMap+hstart*input_w_add_pad+wstart+1],maxIdx);
					maxIdx=max(input[offsetOfFeatMap+(hstart+1)*input_w_add_pad+wstart],maxIdx);
					maxIdx=max(input[offsetOfFeatMap+(hstart+1)*input_w_add_pad+wstart+1],maxIdx);
					top[pool_index]=maxIdx;
				}
			}
		}
	}
	delete [] input;
}

float CNN::ReLU(float data){
	return data>0 ? data : 0;
}

void CNN::meanSubtractAndScale(const unsigned char * image, const int input_image_size, string tName){
	int idx = layernames_[tName];
	dataParams  &data_params = paramsBin_[idx].dParams;
	float *top = top_[idx];
	int paramsIdx = name2idx_[idx];
	CHECK_INFO(paramsIdx >= 0, "error");
	float *mean = params_[paramsIdx][0];
	CHECK_INFO(input_image_size == data_params.channels*data_params.h*data_params.w, "mean file not matched!");
	for(int i=0; i < input_image_size; ++i){
		top[i] = (float(image[i]) - mean[i]) * data_params.scale_factor;
	}
}


void Qsort(pair<int, float> *pair_array, int low, int high)
{
    if(low >= high)
    {
        return;
    }
    int first = low;
    int last = high;
    pair<int, float> key = pair_array[first];
 
    while(first < last)
    {
        while(first < last && pair_array[last].second <= key.second)
        {
            --last;
        }
 
        pair_array[first] = pair_array[last];
 
        while(first < last && pair_array[first].second >= key.second)
        {
            ++first;
        }
         
        pair_array[last] = pair_array[first];    
    }
    pair_array[first] = key;
    Qsort(pair_array, low, first-1);
    Qsort(pair_array, first+1, high);
}


//argMax of input
 inline int* CNN::judge(const float *input, int len){

 	for(int i=0;i<len;i++){
 		result_pair_array[i] = make_pair(i,input[i]);
 	}
 	//Use Quick sort method to sort the array
 	Qsort(result_pair_array, 0,len-1);
 	for(int i=0;i<len;i++){
 		result_array[i]= result_pair_array[i].first;
 	}

 	float maxVal =  input[0];
	int maxIdx = 0;
	for(int i = 1; i < len; ++i){
		if( input[i] > maxVal){
			maxVal = input[i];
			maxIdx = i;
		}
	}
	
	CHECK_INFO(maxIdx == result_array[0], "error");
	return result_array;
}


void CNN::fc_template_sparse_mv(const string bName, const string tName, bool isRelu){
	float *bottom = top_[layernames_[bName]];
	
	int idx = layernames_[tName];
	fcParams  &fcParams_ = paramsBin_[idx].fParams;
	float *top = top_[idx];

	int paramsIdx = name2idx_[layernames_[tName]];
	CHECK_INFO(paramsIdx >= 0, "error");
	float *a = params_[paramsIdx][0];
	float *bias = params_[paramsIdx][1];
	int *ia = (int*)params_[paramsIdx][2];
	int *ja = (int*)params_[paramsIdx][3];
	memset(top, 0, sizeof(float) * fcParams_.outputSize);
	/************************************************************************************************************/
	/*  void mkl_cspblas_scsrgemv (char *transa, int *m, float *a, int *ia, int *ja, float *x, float *y);		*/
	/*	transa: If transa = 'N' or 'n', then the matrix-vector product is computed as y := A*x					*/
    /*			If transa = 'T' or 't' or 'C' or 'c', then the matrix-vector product is computed as y := A'*x.	*/
	/*																											*/
	/*	m:		INTEGER. Number of rows of the matrix A.														*/
	/*																											*/
	/*	a:		REAL for mkl_cspblas_scsrgemv.																	*/
	/*			Array containing non-zero elements of the matrix A. 											*/
	/*			Its length is equal to the number of non-zero elements in the matrix A.							*/
	/*																											*/
	/*	ia:		INTEGER. Array of length m + 1, containing indices of elements in the array a, 					*/
	/*			such that ia(I) is the index in the array a of the first non-zero element from the row I. 		*/
	/*			The value of the last element ia(m) is equal to the number of non-zeros. 						*/
	/*																											*/
	/*	ja:		INTEGER. Array containing the column indices for each non-zero element of the matrix A.			*/
	/*			Its length is equal to the length of the array a. 												*/
	/*																											*/
	/*	x:		REAL for mkl_cspblas_scsrgemv.																	*/
	/*			Array, size is m.																				*/
	/*			One entry, the array x must contain the vector x.												*/
	/************************************************************************************************************/
	const char TransA = 'N';
	mkl_cspblas_scsrgemv(&TransA, &fcParams_.outputSize, a, ia, ja, bottom, top);

	for (int t = 0; t < fcParams_.outputSize; ++t)
	{
		top[t] += bias[t];
		if (isRelu)
		{
			top[t] = ReLU(top[t]);
		}
	}
}

void CNN::fc_template_BN_PReLU_sparse_mv(const string bName, const string tName,const string bnName,const string preluName){
	float *bottom = top_[layernames_[bName]];
	
	int idx = layernames_[tName];
	fcParams  &fcParams_ = paramsBin_[idx].fParams;
	float *top = top_[idx];

	int paramsIdx = name2idx_[layernames_[tName]];
	CHECK_INFO(paramsIdx >= 0, "error");
	float *a = params_[paramsIdx][0];
	float *bias = params_[paramsIdx][1];
	int *ia = (int*)params_[paramsIdx][2];
	int *ja = (int*)params_[paramsIdx][3];

	//read bn params
	idx = layernames_[bnName];
	batchNormParams  &bn_params = paramsBin_[idx].bnParams;
	paramsIdx = name2idx_[idx];
	CHECK_INFO(paramsIdx >= 0, "error");
	float *mean = params_[paramsIdx][0];
	float *var = params_[paramsIdx][1];
	//read prelu params
	idx = layernames_[preluName];
	PReLUParams  &prelu_params = paramsBin_[idx].preluParams;
	paramsIdx = name2idx_[idx];
	CHECK_INFO(paramsIdx >= 0, "error");
	float *prelu_weight = params_[paramsIdx][0];

	memset(top, 0, sizeof(int) * fcParams_.outputSize);

	const char TransA = 'N';
	mkl_cspblas_scsrgemv(&TransA, &fcParams_.outputSize, a, ia, ja, bottom, top);

	for (int t = 0; t < fcParams_.outputSize; ++t)
	{
		top[t] +=mean[t];
		top[t] *=var[t];
		// prelu
		top[t]= max(top[t], 0) +  prelu_weight[t] *min(top[t], 0);
	}
}

int* CNN::forward(unsigned char* data,const int dataSize){

	meanSubtractAndScale(data,dataSize,"data");

	conv_BLAS_template_BN_PReLU("data", "conv1", "bn1", "PReLU1");	//Computation of convolution layer combined with batchNorm and PReLU	
	max_pooling_extend_loop_k3("conv1", "pool1");					//Computation of max-pooling layer with kernel size of 3, stride of 1

	conv_BLAS_template("pool1", "conv2_1");							//Computation of convolution layer
	conv_BLAS_template_BN_PReLU("conv2_1", "conv2_2", "bn2_2", "PReLU2_2");
	max_pooling_extend_loop_k3("conv2_2", "pool2");

	conv_BLAS_template("pool2", "conv3_1");
	conv_BLAS_template_BN_PReLU("conv3_1", "conv3_2", "bn3_2", "PReLU3_2");
	max_pooling_extend_loop_k3("conv3_2", "pool3");

	conv_BLAS_template("pool3", "conv4_1_1");
	conv_BLAS_template_BN_PReLU("conv4_1_1", "conv4_1_2", "bn4_1_2", "PReLU4_1_2");
	conv_BLAS_template("conv4_1_2", "conv4_2_1");
	conv_BLAS_template_BN_PReLU("conv4_2_1", "conv4_2_2", "bn4_2_2", "PReLU4_2_2");
	max_pooling_extend_loop_k3("conv4_2_2", "pool4");

	conv_BLAS_template("pool4", "conv5_1_1");
	conv_BLAS_template_BN_PReLU("conv5_1_1", "conv5_1_2", "bn5_1_2", "PReLU5_1_2");
	conv_BLAS_template("conv5_1_2", "conv5_2_1");
	conv_BLAS_template_BN_PReLU("conv5_2_1", "conv5_2_2", "bn5_2_2", "PReLU5_2_2");
	max_pooling_extend_loop_k3("conv5_2_2", "pool5");

	fc_template_BN_PReLU_sparse_mv("pool5", "fc1", "bn6", "PReLU6");	//Sparse_mv Computation of full-connect layer combiend with bacthNorm and PReLU 
	fc_template_sparse_mv("fc1", "fc2",false);							//Sparse_mv Computation of full-connect layer

	return judge(top_[layernames_["fc2"]], paramsBin_[layernames_["fc2"]].fParams.outputSize);
}

//loading params from compress binary file
void CNN::LoadParams(const string paramsFileAddr){
	FILE *fin;
	fin = fopen(paramsFileAddr.c_str(), "rb");
	if(fin == NULL){
		printf("Cannot open %s\n", paramsFileAddr.c_str());
		exit(1);
	}
	printf("\n\nLoading network parameters from %s...\n", paramsFileAddr.c_str());

	int idx = 0;
	int num;
	char name[64];
	char type[64];
	int fread_temp;

	while(!feof(fin)){
		//read layer name and layer type
		fread_temp = fread(&num, sizeof(int), 1, fin);
		if(fread_temp == 0) break;
		fread_temp = fread(&name, sizeof(char), num, fin);
		name[num] = '\0';
		layernames_.insert(map<string, int>::value_type(name, idx));
		fread_temp = fread(&num, sizeof(int), 1, fin);
		fread_temp = fread(&type, sizeof(char), num, fin);
		type[num] = '\0';
		layertype_.push_back(type);

		//load LayerParams to vector<ParamsBin> paramsBin_; load mean, weight, bias and var to vector<vector<float*> > params_.
		if(!strcmp(type, "Data")){
			ParamsBin attributes;
			fread_temp = fread(&attributes.dParams, sizeof(dataParams), 1, fin);
			paramsBin_.push_back(attributes);
			int image_size = attributes.dParams.channels * attributes.dParams.h * attributes.dParams.w;
			float* input_mean = new float[image_size];
			fread_temp = fread(input_mean, sizeof(float), image_size, fin);		//load mean value
			vector<float*> tmp;
			tmp.push_back(input_mean);
			name2idx_.push_back(params_.size());
			params_.push_back(tmp);
			float *data = new float[image_size];
			top_.push_back(data);
		}
		else if(!strcmp(type, "Convolution")){
			ParamsBin attributes;
			fread_temp = fread(&attributes.cParams, sizeof(convParams), 1, fin);
			paramsBin_.push_back(attributes);
			int weightSize = attributes.cParams.outputChannels * attributes.cParams.inputChannels * attributes.cParams.kernelH * attributes.cParams.kernelW;
			int biasSize = attributes.cParams.outputChannels;
			float* weights = new float[weightSize];
			memset(weights,0,sizeof(float)*weightSize);
			float* bias = new float[biasSize];
			vector<float*> tmp;

			//the weights of conv1 are not compressed, so it should not be unpacked;
			//the weights of other convLayers need to be unpacked
			if(!strcmp(name, "conv1")){
				fread_temp = fread(weights, sizeof(float), weightSize, fin);				
				tmp.push_back(weights);
			}
			else 
			{
		 		printf("layer name : %s\t",name);
				CHECK_INFO(unpack_cluster_blob(weights, fin, weightSize),"unpack fail"); // unpack the conv params and check
				tmp.push_back(weights);
			}
			fread_temp = fread(bias, sizeof(float), biasSize, fin);
			tmp.push_back(bias);
			name2idx_.push_back(params_.size());
			params_.push_back(tmp);
			float *data = new float[attributes.cParams.outputChannels * attributes.cParams.outputH * attributes.cParams.outputW];
			top_.push_back(data);
		}
		else if(!strcmp(type, "InnerProduct")){
		 	printf("layer name : %s\t",name);
			ParamsBin attributes;
			fread_temp = fread(&attributes.fParams, sizeof(fcParams), 1, fin);
			paramsBin_.push_back(attributes);
			int weightSize = attributes.fParams.inputSize * attributes.fParams.outputSize;
			int biasSize = attributes.fParams.outputSize;
			float* weights = new float[weightSize];
			memset(weights, 0, sizeof(float) * weightSize);
			float* bias = new float[biasSize];

			//unpack the weights of InnerProduct and check
			CHECK_INFO(unpack_cluster_blob(weights, fin, weightSize),"unpack fail");

			//compute the CSR format of the weights
			vector<int> non_zero_idx;
			int length = count_non_zero_num(weights, non_zero_idx, weightSize);
			float* a = new float[length];
			int* ia = new int[attributes.fParams.outputSize + 1];
			int* ja = new int[length];
			CSR_coding(weights, a, ia, ja, non_zero_idx, attributes.fParams.outputSize, attributes.fParams.inputSize);

			vector<float*> tmp;
			fread_temp = fread(bias, sizeof(float), biasSize, fin);
			tmp.push_back(a);
			tmp.push_back(bias);
			tmp.push_back((float*)ia);
			tmp.push_back((float*)ja);

			name2idx_.push_back(params_.size());
			params_.push_back(tmp);
			float *data = new float[attributes.fParams.outputSize];
			top_.push_back(data);
		}
		else if(!strcmp(type, "BatchNorm")){
			ParamsBin attributes;
			vector<float*> tmp;
			fread_temp = fread(&attributes.bnParams, sizeof(batchNormParams), 1, fin);
			paramsBin_.push_back(attributes);
			int paramsSize = attributes.bnParams.channels;
			float* mean = new float[paramsSize];
			float* var = new float[paramsSize];
			float* mean_extend;
			fread_temp = fread(mean, sizeof(float), paramsSize, fin);
			fread_temp = fread(var, sizeof(float), paramsSize, fin);
			//combine mean and bias to reduce calculation
			int idx = layernames_[attributes.bnParams.bottom_layer_name];
			float *bias;
			if(!strcmp(layertype_[idx].c_str(), "Convolution")){
				convParams  &conv_params = paramsBin_[idx].cParams;
				CHECK_INFO(conv_params.outputChannels== paramsSize, "error");
				int paramsIdx = name2idx_[idx];
				bias = params_[paramsIdx][1];

				const int output_map_size = conv_params.outputH*conv_params.outputW;
				mean_extend = new float[paramsSize*output_map_size]();
				for(int i=0;i<paramsSize;++i){
					mean[i]=bias[i]-mean[i];
					for(int ii=0;ii<output_map_size;ii++){
						mean_extend[ii+i*output_map_size]=mean[i];
					}
				}
				tmp.push_back(mean_extend);
				tmp.push_back(var);
				delete[] mean;
			}
			else if(!strcmp(layertype_[idx].c_str(), "InnerProduct")){
				fcParams  &fc_params = paramsBin_[idx].fParams;
				CHECK_INFO(fc_params.outputSize== paramsSize, "error");
				int paramsIdx = name2idx_[idx];
				bias = params_[paramsIdx][1];
				for(int i=0;i<paramsSize;++i){
					mean[i]=bias[i]-mean[i];
				}
				tmp.push_back(mean);
				tmp.push_back(var);
			}
			else{
				printf("bn bottom layer error\n");
				CHECK_INFO(0, "error");
			}

			name2idx_.push_back(params_.size());
			params_.push_back(tmp);
			top_.push_back(NULL);

		}
		else if(!strcmp(type, "PReLU")){
			ParamsBin attributes;
			fread_temp = fread(&attributes.preluParams, sizeof(PReLUParams), 1, fin);
			paramsBin_.push_back(attributes);
			int paramsSize = attributes.preluParams.channels;
			float* weight = new float[paramsSize];
			fread_temp = fread(weight, sizeof(float), paramsSize, fin);
			vector<float*> tmp;
			tmp.push_back(weight);
			name2idx_.push_back(params_.size());
			params_.push_back(tmp);
			top_.push_back(NULL);
		}
		else if(!strcmp(type, "Pooling")){
			ParamsBin attributes;
			fread_temp = fread(&attributes.pParams, sizeof(poolParams), 1, fin);
			paramsBin_.push_back(attributes);
			name2idx_.push_back(-1);
			float *data = new float[attributes.pParams.channels * attributes.pParams.outputH * attributes.pParams.outputW];
			top_.push_back(data);
		}
		else{
			printf("ignore %s\n", name);
		}
		idx++;
	}
}

// count the num of non-zero weights
int count_non_zero_num(float* weights, vector<int>& non_zero_idx, int weightSize){
	int count_length=0;

	for(int i=0;i<weightSize;++i){
		if(weights[i]!=0){
			count_length++;
			non_zero_idx.push_back(i);
		}
	}
	return count_length;
}

void CSR_coding(float* A, float* a, int* ia, int* ja, vector<int> &non_zero_idx, const int M, const int N){
	int j = 0;
	for(int i = 0; i < non_zero_idx.size(); i++){
		a[i] = A[non_zero_idx[i]];
		ja[i] = non_zero_idx[i] % N;
		
		if(non_zero_idx[i] / N >= j){
			ia[j] = i;
			j++;
		}
	}
	if(j != M){
		cout << "one row in the matrix A is all zeros!" << endl;
	}

	ia[j] = non_zero_idx.size();
}

bool unpack_cluster_blob(float* weights, FILE* &fp, int weightSize){
	extern int sum_weight_size;
	extern int sum_zero_num;
	extern int sum_compress_byte;

	int count_weight_num; 
	int save_diff_index_byte_size;
	int index_bit ;
	byte* save_byte;
	byte *array_diff;
	int* array_cluster_label_index;
	int diff_index_pair_length;
	int fread_temp = 0;
	int compress_byte = 0;

	unsigned short count_cluster_center;
	fread_temp = fread(&count_cluster_center, sizeof(unsigned short),1,fp);
	const int count_cluster_center_const =count_cluster_center;
	compress_byte += 2;

	float* cluster_center =new float[count_cluster_center_const]();
	fread_temp = fread(cluster_center, sizeof(float),count_cluster_center_const,fp);
	compress_byte += 4*count_cluster_center_const;

	const int weight_bit = get_log_two(count_cluster_center);
	
	fread_temp = fread(&count_weight_num, sizeof(int),1,fp);	
	CHECK_INFO(count_weight_num == weightSize,"don't match");
	compress_byte += 4;

	fread_temp = fread(&save_diff_index_byte_size, sizeof(int),1,fp);
	fread_temp = fread(&index_bit, sizeof(unsigned char),1,fp);
	fread_temp = fread(&diff_index_pair_length, sizeof(int),1,fp);
	const int diff_index_pair_length_const = diff_index_pair_length;
	compress_byte += 4 + 1 + 4;

	array_diff =new byte[diff_index_pair_length_const]() ;
    array_cluster_label_index = new int[diff_index_pair_length_const]();

	save_byte = new byte[save_diff_index_byte_size]();
	fread_temp=fread(save_byte, sizeof(byte),save_diff_index_byte_size,fp);
	compress_byte += save_diff_index_byte_size;

	for(int nStart = 0,nEnd=0,byte_temp=0,i=0 ;i<diff_index_pair_length_const;i++){
		ReadDataFromBuffer(save_byte,nStart,index_bit,nEnd,byte_temp);
		array_diff[i]=byte_temp;
		nStart=nEnd;
		ReadDataFromBuffer(save_byte,nStart,weight_bit,nEnd,byte_temp);
		array_cluster_label_index[i]=byte_temp;
		nStart=nEnd;
	}

	reconvery_blob_data(weights,cluster_center,array_cluster_label_index, diff_index_pair_length,count_weight_num, array_diff);

	int zero_num = 0;
	for (int i = 0; i < count_weight_num; ++i)
	{
		if (weights[i] == 0)
		{
			zero_num++;
		}
	}
	printf("params num = %d\t", count_weight_num);
	printf("pR = %.2f%%\t",(1 - zero_num*1.0/count_weight_num)* 100);
	printf("p+c = %.2f%%\n", compress_byte*1.0/4.0/count_weight_num * 100);

	sum_weight_size += count_weight_num;
	sum_zero_num += zero_num;
	sum_compress_byte += compress_byte;

	delete[] array_diff;
	delete[] array_cluster_label_index;
	delete[] save_byte;
	return true;
}

void reconvery_blob_data(float* weights,const float* cluster_center,const int* array_cluster_label_index,const int diff_index_pair_length,\
	const int weights_num,const byte* array_diff){
	
	int count_blob_index=0,index_pair = 0;
	count_blob_index=array_diff[0];
	weights[count_blob_index]=cluster_center[array_cluster_label_index[0]];
	index_pair++;

	for(;index_pair<diff_index_pair_length;++index_pair){
		count_blob_index+=(array_diff[index_pair]+1);
		weights[count_blob_index]=cluster_center[array_cluster_label_index[index_pair]];
	}
}

