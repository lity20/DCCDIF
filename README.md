# DCC-DIF

This is an implementation of the CVPR 2022 paper [Learning Deep Implicit Functions for 3D Shapes with Dynamic Code Clouds](https://arxiv.org/abs/2203.14048).

[[Paper](https://yushen-liu.github.io/main/pdf/LiuYS_CVPR2022_DCC-DIF.pdf)] [[Supplement](https://yushen-liu.github.io/main/pdf/LiuYS_CVPR2022_DCC-DIF-supp.pdf)] [[Data](https://drive.google.com/drive/folders/1UIh8AxULPi9vY0c1wuZ7HRX4XxDI6a1P?usp=sharing)] [[Project page](https://lity20.github.io/DCCDIF_project_page/)]

## Data

We use the [ShapeNet](https://shapenet.org/) dataset in our experiments. To run our method, the sampled points and their signed distances are needed as training data. We put the code and documents about data processing in [`sample_SDF_points.zip`](https://drive.google.com/drive/folders/1UIh8AxULPi9vY0c1wuZ7HRX4XxDI6a1P). For ease of use, we also provide the processed data of `bench` in [`02828884_sdf_samples.zip`](https://drive.google.com/drive/folders/1UIh8AxULPi9vY0c1wuZ7HRX4XxDI6a1P). 

## Running code

First, install python denpendencies:

```
pip install -r requirements.txt
```

Then, prepare your configuration file based the example we provide in `configs/bench.py`. You may need to specify the paths to data and split files.


Now you can reproduce the experimental results in our paper by running:
```
python train.py configs.bench
python reconstruct.py configs.bench
python evaluate.py configs.bench
```

The pretrained models can be found [here](https://drive.google.com/drive/folders/1UIh8AxULPi9vY0c1wuZ7HRX4XxDI6a1P), we also provide demo code and document to use pretrained models in [`usage_demo.zip`](https://drive.google.com/drive/folders/1UIh8AxULPi9vY0c1wuZ7HRX4XxDI6a1P).

## Citing DCC-DIF

If you find this code useful, please consider citing:
```
@inproceedings{Li2022DCCDIF,
    title={Learning Deep Implicit Functions for 3D Shapes with Dynamic Code Clouds},
    author={Tianyang Li and Xin Wen and Yu-Shen Liu and Hua Su and Zhizhong Han},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2022}
}
```
