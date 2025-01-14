## Spectral Heterogeneous Graph Convolutions via Positive Noncommutative Polynomials
This repository contains a PyTorch implementation of our WWW2024 paper paper "[Spectral Heterogeneous Graph Convolutions via Positive Noncommutative Polynomials](https://arxiv.org/abs/2305.19872)." 
## Environment Settings
- pytorch 1.12.1
- numpy 1.23.1
- dgl 0.9.1
- torch-geometric 2.1.0
- tqdm 4.64.1
- scipy 1.9.3
- seaborn 0.12.0
- scikit-learn 1.1.3
- ogb 1.3.6

## Datasets
For HGB datasets, you can download them from the Heterogeneous Graph Benchmark ([HGB](https://github.com/THUDM/HGB) ). Ogbn-mag can be downloaded automatically.

## Code Structure
The folder "hgb" is the code for the node classification on HGB; The folder "ogbn-mag" is the code for the node classification on ogbn-mag; The folder "hgb-lp" is the code for the link prediction on HGB.

## Citation
```sh
@inproceedings{pshgcn,
author = {He, Mingguo and Wei, Zhewei and Feng, Shikun and Huang, Zhengjie and Li, Weibin and Sun, Yu and Yu, Dianhai},
title = {Spectral Heterogeneous Graph Convolutions via Positive Noncommutative Polynomials},
year = {2024},
isbn = {9798400701719},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3589334.3645515},
doi = {10.1145/3589334.3645515},
booktitle = {Proceedings of the ACM Web Conference 2024},
pages = {685–696},
numpages = {12},
keywords = {heterogeneous graph neural networks, positive noncommutative polynomials, spectral graph convolutions},
location = {Singapore, Singapore},
series = {WWW '24}
}
```

## Contact
If you have any questions, please feel free to contact me with mingguo@ruc.edu.cn