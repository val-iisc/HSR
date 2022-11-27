# Hierarchical Semantic Regularization of Latent Spaces in StyleGANs

[Project Page](https://sites.google.com/view/hsr-eccv22/)

## Requirements

* 64-bit Python 3.8 and PyTorch 1.8.0 (or later). See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions.
* CUDA toolkit 11.0 or later.
* python libraries: see scripts/requirements.txt
* StyleGAN2 code relies heavily on custom PyTorch extensions. For detail please refer to the repo [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)

To setup conda env with all requirements and pretrained networks run the following command:
```.bash
conda create -n hsr python=3.8
conda activate hsr
git clone https://github.com/val-iisc/hsr.git
cd hsr
bash docs/setup.sh
```

## Inference

**To generate images**: 


```.bash
# random image generation from LSUN Church model

python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 --network=<path to network>
```
The above command generates 4 images using the provided seed values and saves it in `out` directory controlled by `--outdir`. Our generator architecture is same as styleGAN2 and can be similarly used in the Python code as described in [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/README.md#using-networks-from-python).

**Evaluation**:
```.bash
python calc_metrics.py --network <path to network> --metrics ppl2_wend,fid50k_full,pr50k3_full --data ./datasets/ffhq256x256.zip
```

**Pretrained Models**:
We provide pretrained models for the folowing datasets, trained at 256x256 resolution.
[FFHQ](https://huggingface.co/tejank10/HSR/resolve/main/ffhq256.pkl), [LSUN-Church](https://huggingface.co/tejank10/HSR/resolve/main/church256.pkl)

## Datasets

Dataset preparation is same as given in [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/README.md#preparing-datasets).
Example setup for 100-shot AnimalFace Dog and LSUN Church

**AnimalFace Dog**
```.bash
mkdir datasets
wget https://data-efficient-gans.mit.edu/datasets/AnimalFace-dog.zip -P datasets
```

**LSUN Church**
```.bash
cd ..
git clone https://github.com/fyu/lsun.git
cd lsun
python3 download.py -c church_outdoor
unzip church_outdoor_train_lmdb.zip
cd ../hsr
mkdir datasets
python dataset_tool.py --source ../lsun/church_outdoor_train_lmdb/ --dest datasets/church.zip --transform=center-crop --width=256 --height=256
```

All other datasets can be downloaded from their repsective websites:

[FFHQ](https://github.com/NVlabs/ffhq-dataset), [LSUN Categories](http://dl.yf.io/lsun/objects/), [AFHQ](https://github.com/clovaai/stargan-v2), [AnimalFace Dog](https://data-efficient-gans.mit.edu/datasets/AnimalFace-dog.zip), [AnimalFace Cat](https://data-efficient-gans.mit.edu/datasets/AnimalFace-cat.zip)

## Training
```.bash
python train_baseline.py --outdir ./models --data ../datasets/ffhq256x256.zip --cfg paper256 --mirror 1 --aug noaug --batch 16 --gpus 2 --kimg 500
```
```.bash
python train_hsr.py --outdir ./models --data ../datasets/ffhq256x256.zip --cfg paper256 --mirror 1 --aug noaug --batch 16 --gpus 2 --resume ./models/<path to baseline model directory>/network-snapshot-000500.pkl --model-name DINO
```

## Citation

```
@InProceedings{karmali2022hierarchical,
author="Karmali, Tejan
and Parihar, Rishubh
and Agrawal, Susmit
and Rangwani, Harsh
and Jampani, Varun
and Singh, Maneesh
and Babu, R. Venkatesh",
title="Hierarchical Semantic Regularization of Latent Spaces in StyleGANs",
booktitle="Computer Vision -- ECCV 2022",
year="2022",
pages="443--459",
}


```

## Acknowledgments
Our codebase is built on [vision-aided-gan](https://github.com/nupurkmr9/vision-aided-gan).
