## PSHGCN on HGB datasets for link prediction

You can run the following commands directly in this folder.
```sh
python training.py --dataset LastFM
python training_dis.py --dataset amazon
```
Note: Reproducing this link prediction task is challenging, as I find the same parameters can yield widely varying results when run on different machines. The code provides the parameters for PSHGCN on LastFM and Amazon, which correspond to the results presented in the paper and were obtained on a specific machine. You may also adjust the Dropout rate and K as needed. Additionally, fixing the positive sample splits and designing a more consistent experimental setup could be worthwhile avenues for further exploration.


## Acknowledgment
This repository benefit from [HGB](https://github.com/THUDM/HGB/tree/master).