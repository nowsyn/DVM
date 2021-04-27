#  Deep Video Matting via Spatio-Temporal Alignment and Aggregation [CVPR2021]
---

Paper: [https://arxiv.org/abs/2104.11208](https://arxiv.org/abs/2104.11208)

### Introduction

Despite the significant progress made by deep learning in natural image matting, there has been so far no representative work on deep learning for video matting due to the inherent technical challenges in reasoning temporal domain and lack of large-scale video matting datasets. In this paper, we propose a deep learning-based video matting framework which employs a novel and effective spatio-temporal feature aggregation module (ST-FAM). As optical flow estimation can be very unreliable within matting regions, ST-FAM is designed to effectively align and aggregate information across different spatial scales and temporal frames within the network decoder. To eliminate frame-by-frame trimap annotations, a lightweight interactive trimap propagation network is also introduced. The other contribution consists of a large-scale video matting dataset with groundtruth alpha mattes for quantitative evaluation and real-world high-resolution videos with trimaps for qualitative evaluation. Quantitative and qualitative experimental results show that our framework significantly outperforms conventional video matting and deep image matting methods applied to video in presence of multi-frame temporal information.

### Framework
<img src="./figures/framework.png" width="800" alt="framework" align=center/>


### Dataset

We composite foreground images and videos onto high-resolution background videos to generate large-scale video matting training/testing dataset. Follow the steps to prepare the datasets. The structure is as the following.

```
DVM
  ├── fg
    ├── image
      ├── train
        ├── alpha
          ├── xxx.png
          ├── yyy.png
          ├── ...
        ├── fg
          ├── xxx.png
          ├── yyy.png
          ├── ...
      ├── test
        ├── alpha
          ├── xxx.png
          ├── yyy.png
          ├── ...
        ├── fg
          ├── xxx.png
          ├── yyy.png
          ├── ...
        ├── trimap
          ├── xxx.png
          ├── yyy.png
          ├── ...
    ├── video
      ├── train
        ├── 0000
          ├── a.mp4
          ├── f.mp4
        ├── ...
      ├── test
        ├── 0000
          ├── a.mp4
          ├── f.mp4
        ├── ...
  ├── bg
    ├── train
      ├── 0000.mp4
      ├── 0001.mp4
      ├── ...
    ├── test
      ├── 0000.mp4
      ├── 0001.mp4
      ├── ...
```

 

1. Please contact Brian Price (bprice@adobe.com) for the Adobe Image Matting dataset.

2. Put training fg/alpha images and testing fg/alpha/trimap images from Adobe dataset in the corresponding directories.

3. Download training/testing videos and place them in the corresponding directories. 

   Link: https://pan.baidu.com/s/1yBJr0SqsEjDToVAUb8dSCw  Password: l9ck

4. Run the command to generate training/testing datasets. About 1T storage is needed.

   ```
   bash run.sh
   ```


### Reference

If you find our work useful in your research, please consider citing:

```
@article{sun2021dvm,
  author    = {Yanan Sun and Guanzhi Wang and Qiao Gu and Chi-Keung Tang and Yu-Wing Tai}
  title     = {Deep Video Matting via Spatio-Temporal Alignment and Aggregation},
  journal   = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year      = {2021},
}
```



### Contact

If you have any questions or suggestions about this repo, please feel free to contact me ([now.syn@gmail.com](mailto:now.syn@gmail.com)).
