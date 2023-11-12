# WinCLIP
This is an unofficial implementation of [WinCLIP](https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf) in [AnomalyCLIP](https://arxiv.org/abs/2310.18961)

The implementation of CLIP is based on [open_clip](https://github.com/mlfoundations/open_clip)

  
## Performance evaluation
### Zero-shot
#### [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)
| objects    |   auroc_px |   f1_px |   ap_px |   aupro |   auroc_sp |   f1_sp |   ap_sp |
|:-----------|-----------:|--------:|--------:|--------:|-----------:|--------:|--------:|
| carpet     |       90.9 |    33.9 |    26   |    66.3 |       99.3 |    97.8 |    99.8 |
| bottle     |       85.7 |    49.4 |    49.8 |    69.9 |       98.6 |    97.6 |    99.5 |
| hazelnut   |       95.7 |    39.1 |    33.3 |    81.3 |       92.3 |    88.6 |    96   |
| leather    |       95.5 |    30.8 |    20.5 |    86   |      100   |   100   |   100   |
| cable      |       61.3 |    12.2 |     6.2 |    39.4 |       85   |    84.8 |    89.8 |
| capsule    |       87   |    14.3 |     8.6 |    63.8 |       68.7 |    93.5 |    90.5 |
| grid       |       79.4 |    13.7 |     5.7 |    49.3 |       99.2 |    98.2 |    99.7 |
| pill       |       72.7 |    11.8 |     7   |    66.9 |       81.5 |    91.6 |    96.4 |
| transistor |       83.7 |    27   |    20.2 |    45.5 |       89.1 |    80   |    84.9 |
| metal_nut  |       49.3 |    23.8 |    10.8 |    39.7 |       96.2 |    95.3 |    99.1 |
| screw      |       91.1 |    11.3 |     5.4 |    70.2 |       71.7 |    85.9 |    87.7 |
| toothbrush |       86.2 |    10.5 |     5.5 |    67.9 |       85.3 |    88.9 |    94.5 |
| zipper     |       91.7 |    27.8 |    19.4 |    72   |       91.2 |    93.4 |    97.5 |
| tile       |       79.1 |    30.8 |    21.2 |    54.5 |       99.9 |    99.4 |   100   |
| wood       |       85.1 |    35.4 |    32.9 |    56.3 |       97.6 |    95.2 |    99.3 |
| mean       |       82.3 |    24.8 |    18.2 |    61.9 |       90.4 |    92.7 |    95.6 |

#### VisA
| objects    |   auroc_px |   f1_px |   ap_px |   aupro |   auroc_sp |   f1_sp |   ap_sp |
|:-----------|-----------:|--------:|--------:|--------:|-----------:|--------:|--------:|
| candle     |       87   |     8.9 |     2.3 |    77.7 |       94.9 |    90.6 |    95.4 |
| capsules   |       80   |     4.2 |     1.4 |    39.4 |       79.4 |    80.5 |    87.9 |
| cashew     |       84.8 |     9.6 |     4.8 |    78.4 |       91.2 |    88.9 |    96   |
| chewinggum |       95.4 |    31.5 |    24   |    69.6 |       95.5 |    93.8 |    98.2 |
| fryum      |       87.7 |    16.2 |    11.1 |    74.4 |       73.6 |    80   |    86.9 |
| macaroni1  |       50.3 |     0.1 |     0   |    24.7 |       79   |    74.2 |    80   |
| macaroni2  |       44.7 |     0.1 |     0   |     8   |       67.1 |    68.8 |    65.1 |
| pcb1       |       38.7 |     0.9 |     0.4 |    20.7 |       72.1 |    70.2 |    73   |
| pcb2       |       58.7 |     1.5 |     0.4 |    20.6 |       47   |    67.1 |    46.1 |
| pcb3       |       76   |     2.1 |     0.7 |    43.7 |       63.9 |    67.6 |    63   |
| pcb4       |       91.4 |    24.6 |    15.5 |    74.5 |       74.2 |    75.7 |    70.1 |
| pipe_fryum |       83.6 |     8.3 |     4.4 |    80.3 |       67.8 |    80.3 |    82.1 |
| mean       |       73.2 |     9   |     5.4 |    51   |       75.5 |    78.2 |    78.7 |
### Few-shot
Soon



## Quick start
Zero-shot anomaly detection 
```sh
bash zero_shot.sh
```
Few-shot anomaly detection 
```sh
bash few_shot.sh
```


## BibTex Citation

If you find this paper and repository useful, please cite our paper.

```
@article{zhou2023anomalyclip,
  title={AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection},
  author={Zhou, Qihang and Pang, Guansong and Tian, Yu and He, Shibo and Chen, Jiming},
  journal={arXiv preprint arXiv:2310.18961},
  year={2023}
}
```
