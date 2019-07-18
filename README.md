# D-LADMM

***********************************************************************************************************

This repository is for Differentiable Linearized ADMM (to appear in ICML 2019)

By Xingyu Xie, [Jianlong Wu](https://jlwu1992.github.io), [Zhisheng Zhong](https://zzs1994.github.io), [Guangcan Liu](http://web2.nuist.edu.cn:8080/jszy/Professor.aspx?id=1990) and [Zhouchen Lin](http://www.cis.pku.edu.cn/faculty/vision/zlin/zlin.htm).


For more details or questions, feel free to contact: 

Xingyu Xie: nuaaxing@gmail.com and Jianlong Wu: jlwu1992@pku.edu.cn

***********************************************************************************************************


### Introduction
we propose Differentiable Linearized ADMM (D-LADMM) for solving the problems with linear constraints. Specifically, D-LADMM is a K-layer
LADMM inspired deep neural network, which is obtained by firstly introducing some learnable weights in the classical Linearized ADMM algorithm and then generalizing the proximal operator to some learnable activation function. Notably, we mathematically prove that there exist a set of learnable parameters for D-LADMM to generate globally converged solutions, and we show that those desired parameters can be attained by training D-LADMM in a proper way. 


### Usage
Here we give a toy example code of D-LADMM for Lena image restoration. An example to run this code:

```Python
python main_lena.py
```

### Some Experiment Results

The testing result of the Lena image restoration. The whole process takes about 30 epochs.

|Model            |Training Loss |PSNR |
|-----------------|------------- |-----|
|D-LADMM (d =  5) |------------- |-----|
|D-LADMM (d = 10) |------------- |-----|
|D-LADMM (d = 15) |------------- |-----|



### Citation

If you find this page useful, please cite our paper:

[[1] Xingyu Xie, Jianlong Wu, Zhisheng Zhong, Guangcan Liu and Zhouchen Lin, Differentiable Linearized ADMM. ICML (2019).](https://arxiv.org/abs/1905.06179])

All rights are reserved by the authors.
