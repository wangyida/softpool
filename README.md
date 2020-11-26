# SoftPoolNet: Shape Descriptor for Point Cloud Completion and Classification

The implementation of our paper accepted in **ECCV** *2020* (*EUROPEAN CONFERENCE ON COMPUTER VISION*, 16th)
**[Yida Wang](https://wangyida.github.io/#about), David Tan, [Nassir Navab](http://campar.in.tum.de/Main/NassirNavab) and [Federico Tombari](http://campar.in.tum.de/Main/FedericoTombari)**.
If you find this work useful in yourr research, please cite:

```bash
@article{DBLP:journals/corr/abs-2008-07358,
  author    = {Yida Wang and
               David Joseph Tan and
               Nassir Navab and
               Federico Tombari},
  title     = {SoftPoolNet: Shape Descriptor for Point Cloud Completion and Classification},
  journal   = {CoRR},
  volume    = {abs/2008.07358},
  year      = {2020},
  url       = {https://arxiv.org/abs/2008.07358},
  archivePrefix = {arXiv},
  eprint    = {2008.07358},
  timestamp = {Fri, 21 Aug 2020 15:05:50 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2008-07358.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## SoftPoolNet

 <img src="imgs/2min_presentation_softpool.gif" alt="road condition" frameborder="0" style="border:0" >

## Train
Our **SoftPool** operators are provided in both Tensorflow and Pytorch frameworks, we recommend to use the Pytorch version.

### Pytorch
As we have some comparison experiments on [GRNet](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540341.pdf) and [MSN](https://arxiv.org/pdf/1912.00280v1.pdf), we suggest that you compile python libs in `chamfer_pkg`, `emd`, `expansion_penalty` and `extensions`.
You can go towards each folder which includes the mentioend libs by `cd`, then
```python
python setup.py install --user
```
Suppose that GPU 0 is supposed to get used for training
```bash
cd pytorch
CUDA_VISIBLE_DEVICES=0 python3 val.py --n_regions 2 --num_points 2048 --model log/wo-unet_shapenet/network.pth  --dataset shapenet
```
