# D-LADMM

***********************************************************************************************************

This repository is for [Differentiable Linearized ADMM](https://arxiv.org/abs/1905.06179) (to appear in ICML 2019)

By Xingyu Xie, [Jianlong Wu](https://jlwu1992.github.io), [Zhisheng Zhong](https://zzs1994.github.io), [Guangcan Liu](http://web2.nuist.edu.cn:8080/jszy/Professor.aspx?id=1990) and [Zhouchen Lin](http://www.cis.pku.edu.cn/faculty/vision/zlin/zlin.htm).


For more details or questions, feel free to contact: 

Xingyu Xie: nuaaxing@gmail.com and Jianlong Wu: jlwu1992@pku.edu.cn

***********************************************************************************************************
### Table of contents
- [Introduction](#introduction)
- [Usage](#usage)
- [Experiment Results](#Experiment-Results)
***********************************************************************************************************
### Introduction
we propose Differentiable Linearized ADMM (D-LADMM) for solving the problems with linear constraints. Specifically, D-LADMM is a K-layer
LADMM inspired deep neural network, which is obtained by firstly introducing some learnable weights in the classical Linearized ADMM algorithm and then generalizing the proximal operator to some learnable activation function. Notably, we mathematically prove that there exist a set of learnable parameters for D-LADMM to generate globally converged solutions, and we show that those desired parameters can be attained by training D-LADMM in a proper way. 

***********************************************************************************************************
### Usage
Here we give a toy example of D-LADMM for the Lena image denoise. An example to run this code:

```Python
python main_lena.py
```
***********************************************************************************************************
### Experiment Results

The testing result of the Lena image denoise. The whole process takes about 30 epochs.

|Model            |Training Loss |PSNR |
|-----------------|------------- |-----|
|D-LADMM (d =  5) |    1.667     |33.57|
|D-LADMM (d = 10) |    1.663     |35.44|
|D-LADMM (d = 15) |    1.659     |35.61|

d means the depth (the number of layers) of the D-LADMM model.

Denoising results of the Lena image:

<div align=left><img src="https://raw.githubusercontent.com/zzs1994/D-LADMM/master/img/vis.png" width="90%" height="90%">

***********************************************************************************************************
### Citation

If you find this page useful, please cite our paper:

	@inproceedings{xie2019differentiable,
		title={Differentiable Linearized ADMM},
		author={Xingyu Xie and Jianlong Wu and Zhisheng Zhong and Guangcan Liu and Zhouchen Lin},
		booktitle={International Conference on Machine Learning},
		pages={6902--6911},
		year={2019}
	}

All rights are reserved by the authors.
***********************************************************************************************************