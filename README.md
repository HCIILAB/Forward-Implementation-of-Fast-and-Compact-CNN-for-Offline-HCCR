# Forward Implementation of Fast and Compact CNN for Offline HCCR

This project is a forward model of a fast and compact convolutional neural network for offline handwritten Chinese chracter recognition(HCCR). For more information, please see the paper: ["Xuefeng Xiao, Building Fast and Compact Convolutional Neural Networks for Offline Handwritten Chinese Character Recognition, arXiv:1702.07975 [cs.CV]"](https://arxiv.org/abs/1702.07975v1). 


## Project components

Image -- contains 1000 gray-valued character samples for test, which have the size of 96*96.

ImageName_Label.txt -- store the names of the test samples and their corresponding labels.

src -- store the code and the parameters of our model

Makefile -- link the MKL and OpenCV library for code compilation.

run. sh -- the shell script for running the project; note that the OMP_NUM_THREADS should be set to 1 to acquired accurate run-time.


## Usage

Prior to compilation, you need to install [MKL](https://software.intel.com/en-us/intel-mkl) and [OpenCV](http://opencv.org/), and modify Makefile if needed. After that, execute "run. sh" file and perform test.


## Experiment result

Our forward model obtains an accuracy of ***97.09%*** on the *ICDAR 2013 offline HCCR competition datast* and consumes a average run-time of ***9.72ms*** for every sample. The experiment is carried out on a single desktop PC, equipped with an Intel® Core™ i7-6700 CPU @ 3.40GHz × 8, 16GB RAM, ubuntu 14.04 LTS operating system. All experiment are executed in the single-thread mode, without GPU acceleration. The size of our model parameters is only ***2.34MB***.
Owing to the memory limitation, we just offer 1000 images from the competition dataset in our provided project.


## Citation

Please cite our paper if it helps your research:

    @inproceedings{xiao2017building,
      author = {Xuefeng Xiao, Lianwen Jin, Yafeng Yang, Weixin Yang, Jun Sun, Tianhai Chang},
      title = {Building Fast and Compact Convolutional Neural Networks for Offline Handwritten Chinese Character Recognition},
      booktitle = {arXiv:1702.07975 [cs.CV]},
      year = {2017},
    }
