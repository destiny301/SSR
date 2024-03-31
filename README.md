# Transformer-Based Selective Super-resolution for Efficient Image Refinement ([AAAI2024](https://ojs.aaai.org/index.php/AAAI/article/download/28560/29089))

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2312.05803)

## Abstract
Conventional super-resolution methods suffer from two drawbacks: substantial computational cost in upscaling an entire large image, and the introduction of extraneous or potentially detrimental information for downstream computer vision tasks during the refinement of the background. To solve these issues, we propose a novel transformer-based algorithm, Selective Super-Resolution (SSR), which partitions images into non-overlapping tiles, selects tiles of interest at various scales with a pyramid architecture, and exclusively reconstructs these selected tiles with deep features. Experimental results on three datasets demonstrate the efficiency and robust performance of our approach for super-resolution. Compared to the state-of-the-art methods, the FID score is reduced from 26.78 to 10.41 with 40\% reduction in computation cost for the BDD100K dataset.

![image](https://github.com/destiny301/SSR/blob/main/flowchart.png)

## Updates
*03/31/2024*

1. Published version: [AAAI2024](https://ojs.aaai.org/index.php/AAAI/article/download/28560/29089)

## Data
Prepare the data as the following structure:
```shell
root/
├──images/
│  ├── train/
│  │   ├── 000001.jpg
│  │   ├── 000002.jpg
│  │   ├── ......
│  ├── ......
├──masks/
│  ├── val/
│  │   ├── 000001.png
│  │   ├── 000002.png
│  │   ├── ......
```

## Citation
If you use SSR in your research or wish to refer to the results published here, please use the following BibTeX entry. Sincerely appreciate it!
```shell
@inproceedings{zhang2024transformer,
  title={Transformer-Based Selective Super-resolution for Efficient Image Refinement},
  author={Zhang, Tianyi and Kasichainula, Kishore and Zhuo, Yaoxin and Li, Baoxin and Seo, Jae-Sun and Cao, Yu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={7},
  pages={7305--7313},
  year={2024}
}
```

## Simple Start:
```shell
python train.py --pretrain --pyramid --conv --eval --imgsz 256 --patchsz 2 --ckpt /your/TR_checkpoint/root
```

TR module can be pretrained solely with ImageNet dataset (the training of this module doesn't need segmentation or Object Detection labels), and load it when initialize our SSR model