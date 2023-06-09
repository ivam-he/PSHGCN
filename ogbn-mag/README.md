## PSHGCN on ogbn-mag

## Requirements
We utilize the sparse_tools offered by SeHGNN. To install this tool, you can refer to their scripts.
```sh
git clone https://github.com/Yangxc13/sparse_tools.git --depth=1
cd sparse_tools
python setup.py develop
```
## Training without extra embeddings and using a one-stage approach.

```sh
python main.py --stage 1 --bias --runs 5
```

## Training with ComplEx embeddings and using a multi-stage approach
To generate embeddings, you will first need to utilize ComplEx. We have provided a bash command in the ./complEx/ folder. You can follow the instructions provided in [NARS](https://github.com/facebookresearch/NARS) or [SeHGNN](https://github.com/ICT-GIMLab/SeHGNN/tree/master/ogbn) to run the command. For your convenience, we have also made the generated files available in the [drive](https://drive.google.com/drive/folders/1nuqEJPHc0w_orDBXSyZkjkV5wzYqSSuT?usp=drive_link), where you can download them for free. Once downloaded, please place the embedding files in the ./complEx/ folder, and then proceed to run the following scripts.
```sh
python main.py --extra_emb --stage 4 --layers_x 2 --runs 5
```
**Results: Val acc: 59.43 ± 0.15, Test acc: 57.52 ± 0.11**


## Acknowledgment
This repository benefit from [SeHGNN](https://github.com/ICT-GIMLab/SeHGNN/tree/master/ogbn).