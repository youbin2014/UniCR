# UniCR: Universally Approximated Certified Robustness via Randomized Smoothing

This is the official implementation for the ECCV22 paper "[UniCR: Universally Approximated Certified Robustness via Randomized Smoothing](https://arxiv.org/abs/2207.02152)". UniCR is to universally approximate the certified radius via randomized smoothing using any continuous i.i.d. pdf functions for any $\ell_p$ norm perturbation. 

## Pre-requirements:
```
pytorch
pyswarms
statsmodels
tqdm 
```

## To train the smoothed model

```
python train.py cifar10 cifar_resnet110 ./model_saved/ --norm=2 --model_path="./model_saved/CIFAR_standard.tar" --MonteNum=1000 --input_size=3072 --iid=1 --pdf_args=-1 --pdf_args=1.0
```

## To certify the inputs using the pre-defined General Normal distribution against $\ell_2$ perturbations:

```
python certification.py cifar10 cifar_resnet110 ./model_saved/ --norm=2 --model_path="./model_saved/CIFAR_Gaussian[-1.0, 1.0]_best.pth" --MonteNum=500 --input_size=3072 --batch=500 --iid=1 --gpu=0 --save_name=CIFAR10_Gaussian_L2_sigma1.0 --pdf_args=-1 --pdf_args=1.41 --pdf_args=2 --samples_begin=0 --samples_end=10000
```

## To certify the inputs using noise optimization:

```
python noise_optimize.py cifar10 cifar_resnet110 ./model_saved/ --norm=2 --model_path="./model_saved/CIFAR_Gaussian[-1.0, 1.0]_best.pth" --MonteNum=1000 --input_size=3072 --batch=512 --iid=1 --save_name=test --pdf_args=-1 --pdf_args=1.41 --pdf_args=2 --opt_args_low=-1 --opt_args_low=0 --opt_args_low=0 --opt_args_high=-1 --opt_args_high=10 --opt_args_high=10 --opt_args_step=-1 --opt_args_step=0.14 --opt_args_step=0.2 --samples_begin=0 --samples_end=500 --gpu=0
```
