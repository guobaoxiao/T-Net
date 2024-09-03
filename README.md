# Tnet and Tnet++
(TPAMI 2024) PyTorch implementation of Paper "T-Net++: Effective Permutation-Equivariance Network for Two-View Correspondence Pruning"

(ICCV 2021) PyTorch implementation of Paper "T-Net: Effective Permutation-Equivariant Network for Two-View Correspondence Learning"

## Abstract

<img src="./Figure/FIG1.png" width="50%" align="right">
We propose a conceptually novel, flexible, and effective framework (named T-Net++) for the task of two-view correspondence pruning. T-Net++ comprises two unique structures: the "-" structure and the "|" structure. The "-" structure utilizes an iterative learning strategy to process correspondences, while the "|" structure integrates all feature information of the "-" structure and produces inlier weights. Moreover, within the "|" structure, we design a new Local-Global Attention Fusion module to fully exploit valuable information obtained from concatenating features through channel-wise and spatial-wise relationships. Furthermore, we develop a Channel-Spatial Squeeze-and-Excitation module, a modified network backbone that enhances the representation ability of important channels and correspondences through the squeeze-and-excitation operation. T-Net++ not only preserves the permutation-equivariance manner for correspondence pruning, but also gathers rich contextual information, thereby enhancing the effectiveness of the network. Experimental results demonstrate that T-Net++ outperforms other state-of-the-art correspondence pruning methods on various benchmarks and excels in two extended tasks.

# Citing NCMNet
If you find the NCMNet code useful, please consider citing:
```bibtex
@article{xiao2024t,
  title={T-Net++: Effective Permutation-Equivariance Network for Two-View Correspondence Pruning},
  author={Xiao, Guobao and Liu, Xin and Zhong, Zhen and Zhang, Xiaoqin and Ma, Jiayi and Ling, Haibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  pages={1--15},
  publisher={IEEE}
}
```
```bibtex
@inproceedings{zhong2021t,
  title={T-Net: Effective Permutation-Equivariant Network for Two-View Correspondence Learning},
  author={Zhong, Zhen and Xiao, Guobao and Zheng, Linxin and Lu, Yan and Ma, Jiayi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1950--1959},
  year={2021}
}
```

# Data Preparation, Trainng, and Testing
This data preparation code of main datasets, trainng, and testing can be found from [[OANet](https://github.com/zjhthu/OANet)]. If you use the part of code related to data generation, testing, or evaluation, you should cite the paper:
```bibtex
@inproceedings{zhang2019oanet,
  title={Learning Two-View Correspondences and Geometry Using Order-Aware Network},
  author={Zhang, Jiahui and Sun, Dawei and Luo, Zixin and Yao, Anbang and Zhou, Lei and Shen, Tianwei and Chen, Yurong and Quan, Long and Liao, Hongen},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2019}
}
