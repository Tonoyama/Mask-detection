# Mask detection
マスク検出器

## PyramidBox(2018)
シングルショット顔検出器。顔検出の課題の1つとして、小さいためぼやけてしまった顔や部分的に隠れている顔を検出できないことがある。そこで本手法では、多段階の顔の大きさを学習させることで様々な顔を検出することが可能になっている。

> [**PyramidBox: A Context-assisted Single Shot
Face Detector.**](https://arxiv.org/abs/1803.07737)
> 
> Xu Tang, Daniel K. Du, Zeqiang He, Jingtuo Liu
> 
> *[arXiv 1803.07737](https://arxiv.org/abs/1803.07737)*

![](https://i.imgur.com/cg3HcKS.jpg)

![](https://i.imgur.com/Ni71Ajz.jpg)

## 準備

```shell
mkdir result
pip3 install -r requirements.txt
```
