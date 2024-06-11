# On Which Nodes Does GCN Fail? Enhancing GCN From the Node Perspective
This repository contains the reference code for the manuscript ``On Which Nodes Does GCN Fail? Enhancing GCN From the Node Perspective" 
## Dependencies
- CUDA 10.2.89
- python 3.6.8
- pytorch 1.9.0
- pyg 2.0.3

## Usage

- For semi-supervised setting, run the following script
```sh
cd Citation
bash semi.sh
```

- For adversary attack, run the following script
```sh
cd Attack
--cora--
python main.py --dataset cora --epochs 2000 --hidden 8 --concat 1  --lr 0.01 --dropout 0.6 --lam 1.2 --I True --ptb_rate 0.0

--citeseer--
python main.py --dataset citeseer --epochs 2000 --hidden 8 --concat 1  --lr 0.01 --dropout 0.6 --lam 0.0 --ptb_rate 0.0

--pubmed--
python main.py --dataset pubmed --epochs 2000 --hidden 4 --concat 1 --lr 0.02 --dropout 0.5 --tem 0.6 --lam 0.0 --ptb_rate 0.0
```


- Label-Feature Smoothing Alignment Algorithm (locate OOC nodes)
```sh
python pre_train_MLP.py -dataset cora
```

- For generate the DSKN-graph
```sh
python KNN-rewiring.py -dataset cora
```

## Citation
```shell
@inproceedings{
huang2024on,
title={On Which Nodes Does {GCN} Fail? Enhancing {GCN} From the Node Perspective},
author={Jincheng Huang and Jialie Shen and Xiaoshuang Shi and Xiaofeng Zhu},
booktitle={Forty-first International Conference on Machine Learning},
year={2024}
```

