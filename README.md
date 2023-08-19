# Selective Super-Resolution

### prepare data as the following data format:

- root:
    - images                # (.jpg)
        - train
        - val
    - masks                # (.png)
        - train
        - val

### Default settings for training:
python train.py --pretrain --pyramid --conv --eval --imgsz 256 --patchsz 2 --ckpt /your/data/root

### TR module can be pretrained solely with ImageNet dataset (the training of this module doesn't need segmentation or Object Detection labels), and load it when initialize our SSR model