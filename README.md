# Pixel Difference Convolution

This repository contains the PyTorch implementation for 
"Pixel Difference Networks for Efficient Edge Detection" 
by 
[Zhuo Su](https://zhuogege1943.com/homepage/)\*, 
[Wenzhe Liu]()\*, 
[Zitong Yu](https://www.oulu.fi/university/researcher/zitong-yu),
[Dewen Hu](https://dblp.org/pers/h/Hu:Dewen.html), 
[Qing Liao](http://liaoqing.me/),
[Qi Tian](https://scholar.google.com/citations?user=61b6eYkAAAAJ&hl=en),
[Matti Pietikäinen](https://en.wikipedia.org/wiki/Matti_Pietik%C3%A4inen_(academic)) and 
[Li Liu](http://lilyliliu.com/)\*\* 
(\* Authors have equal contributions, \*\* Corresponding author). \[[arXiv](https://arxiv.org/abs/2108.07009), [youtube](https://www.youtube.com/watch?v=jEAh_4wm1UU)\]

The writing style of this code is based on [Dynamic Group Convolution](https://github.com/zhuogege1943/dgc).

If you find something useful from our work, please consider citing [our paper](pdc.bib). 

:rocket: **Updates**:
- `Dec. 29, 2021`: Add functions with vanilla conv components in [models/ops\_theta.py](models/ops_theta.py)
- `Aug. 18, 2021`: Load checkpoints in [trained\_models](trained_models)

## Running environment

Training: Pytorch 1.9 with cuda 10.1 and cudnn 7.5 in an Ubuntu 18.04 system <br>
Evaluation: Matlab 2019a

*Ealier versions may also work~ :)*

## Dataset

We use the links in [RCF Repository](https://github.com/yun-liu/rcf#testing-rcf) (really thanks for that). The augmented BSDS500, PASCAL VOC, and NYUD datasets can be downloaded with:

```bash
wget http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz
wget http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz
wget http://mftp.mmcheng.net/liuyun/rcf/data/NYUD.tar.gz
```

To create BSDS dataset, please follow:

1. create a folder */path/to/BSDS500*, 
2. extract *HED-BSDS.tar.gz* to */path/to/BSDS500/HED-BSDS*,
3. extract *PASCAL.tar.gz* to */path/to/BSDS500/PASCAL*,
4. if you want to evaluate on BSDS500 val set, the val images can be downloaded from [this link](https://drive.google.com/file/d/1q0jdUM9PStWT12o1RgTLOKOXN3Ql5OxS/view?usp=sharing), please extract it to */path/to/BSDS500/HED-BSDS/val*,
5. cp the \**.lst* files in [data/BSDS500/HED-BSDS](data/BSDS500/HED-BSDS) to */path/to/BSDS500/HED-BSDS/*, cp the \**.lst* files in [data/BSDS500](data/BSDS500) to */path/to/BSDS500/*.

To create NYUD dataset, please follow:

1. create a folder */path/to/NYUD*,
2. extract *NYUD.tar.gz* to */path/to/NYUD*,
3. cp the \**.lst* files in [data/NYUD](data/NYUD) to */path/to/NYUD/*.


## Training, and Generating edge maps

Here we provide the scripts for training the models appeared in the paper. For example, we refer to the `PiDiNet` model in Table 5 in the paper as `table5_pidinet`. 


**table5_pidinet**
```bash
# train, the checkpoints will be save in /path/to/table5_pidinet/save_models/ during training
python main.py --model pidinet --config carv4 --sa --dil --resume --iter-size 24 -j 4 --gpu 0 --epochs 20 --lr 0.005 --lr-type multistep --lr-steps 10-16 --wd 1e-4 --savedir /path/to/table5_pidinet --datadir /path/to/BSDS500 --dataset BSDS

# generating edge maps using the original model
python main.py --model pidinet --config carv4 --sa --dil -j 4 --gpu 0 --savedir /path/to/table5_pidinet --datadir /path/to/BSDS500 --dataset BSDS --evaluate /path/to/table5_pidinet/save_models/checkpointxxx.pth

# generating edge maps using the converted model, it should output the same results just like using the original model
# the process will convert pidinet to vanilla cnn, using the saved checkpoint
python main.py --model pidinet_converted --config carv4 --sa --dil -j 4 --gpu 0 --savedir /path/to/table5_pidinet --datadir /path/to/BSDS500 --dataset BSDS --evaluate /path/to/table5_pidinet/save_models/checkpointxxx.pth --evaluate-converted

# test FPS on GPU
python throughput.py --model pidinet_converted --config carv4 --sa --dil -j 1 --gpu 0 --datadir /path/to/BSDS500 --dataset BSDS
```

It is similar for other models, please see detailed scripts in [scripts.sh](scripts.sh).

The performance of some of the models are listed below (click the items to download the checkpoints and training logs). FPS metrics are tested on a NVIDIA RTX 2080 Ti, showing slightly faster than that recorded in the paper (you probably get different FPS records in different runs, but they will not vary too much):

| Model                  | ODS   | OIS   | FPS | Training logs |
|------------------------|-------|-------|-----|---------------|
| [table5_baseline](trained_models/table5_baseline.pth)        | 0.798 | 0.816 | 101 |[log](training_logs/table5_baseline.log) |
| [table5_pidinet](trained_models/table5_pidinet.pth)         | 0.807 | 0.823 | 96  |[log](training_logs/table5_pidinet.log), [running log](training_logs/table6_pidinet_running.log)|
| [table5_pidinet-l](trained_models/table5_pidinet-l.pth)       | 0.800 | 0.815 | 135 |[log](training_logs/table5_pidinet-l.log) |
| [table5_pidinet-small](trained_models/table5_pidinet-small.pth)   | 0.798 | 0.814 | 161 |[log](training_logs/table5_pidinet-small.log) |
| [table5_pidinet-small-l](trained_models/table5_pidinet-small-l.pth) | 0.793 | 0.809 | 225 |[log](training_logs/table5_pidinet-small-l.log) |
| [table5_pidinet-tiny](trained_models/table5_pidinet-tiny.pth)    | 0.789 | 0.806 | 182 |[log](training_logs/table5_pidinet-tiny.log) |
| [table5_pidinet-tiny-l](trained_models/table5_pidinet-tiny-l.pth)  | 0.787 | 0.804 | 253 |[log](training_logs/table5_pidinet-tiny-l.log ) |
| [table6_pidinet](trained_models/table6_pidinet.pth)         | 0.733 | 0.747 | 66  |[log](training_logs/table6_pidinet.log), [running_log](training_logs/table6_pidinet_running.log)|
| [table7_pidinet](trained_models/table7_pidinet.pth)         | 0.818 | 0.824 | 17  |[log](training_logs/table7_pidinet.log), [running_log](training_logs/table7_pidinet_running.log)|

## Evaluation

The matlab code used for evaluation in our experiments can be downloaded in [matlab code for evaluation](https://drive.google.com/file/d/16_aqTaeSiKPwCRMwdnvFXH7b7qYL_pKB/view?usp=sharing).

Possible steps:

1. extract the downloaded file to */path/to/edge_eval_matlab*,
2. change the first few lines (path settings) in *eval_bsds.m*, *eval_nyud.m*, *eval_multicue.m* for evaluating the three datasets respectively,
3. in a terminal, open Matlab like 
```bash
matlab -nosplash -nodisplay -nodesktop

# after entering the Matlab environment, 
>>> eval_bsds
```
4. you could change the number of works in parpool in */path/to/edge_eval_matlab/toolbox.badacost.public/matlab/fevalDistr.m* in line 100. The default value is 16.

For evaluating NYUD, following [RCF](https://openaccess.thecvf.com/content_cvpr_2017/html/Liu_Richer_Convolutional_Features_CVPR_2017_paper.html), we increase the localization tolerance from 0.0075 to 0.011. The Matlab code is based on the following links:

- [HED Implementation](https://github.com/xwjabc/hed)
- [Original HED](https://github.com/s9xie/hed)
- [Piotr's Structured Forest matlab toolbox](https://github.com/pdollar/edges)
- [RCF](https://github.com/yun-liu/rcf)

## PR curves
Please follow [plot-edge-pr-curves](https://github.com/MCG-NKU/plot-edge-pr-curves), files for plotting pr curves of PiDiNet are provided in [pidinet_pr_curves](pidinet_pr_curves).

## Generating edge maps for your own images
```bash
python main.py --model pidinet_converted --config carv4 --sa --dil -j 4 --gpu 0 --savedir /path/to/savedir --datadir /path/to/custom_images --dataset Custom --evaluate /path/to/table5_pidinet/save_models/checkpointxxx.pth --evaluate-converted
```

<div align=center>
<img src="https://user-images.githubusercontent.com/18327074/129970337-bb467a8c-825e-47ee-872c-533f0a5da37a.jpg"><br>
The results of our model look like this. The top image is the messy office table, the bottom image is the peaceful Saimaa lake in southeast of Finland.
</div>

