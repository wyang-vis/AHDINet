# AAAI 2025 AHDINet
Asymmetric Hierarchical Difference-aware Interaction Network for Event-guided Motion Deblurring

# Installation

The model is built in PyTorch 2.0.1 and tested on Ubuntu 20.04.6 environment.

For installing, follow these instructions

    conda create -n pytorch-2.0 
    conda activate pytorch-2.0
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm

Install warmup scheduler

    cd pytorch-gradual-warmup-lr; python setup.py install; cd ..

# Training and Evaluation
## Train
- Download the [GoPro events train dataset](https://pan.baidu.com/s/1lw-CW3QH-ZJdpP0CT9oMnw) and [GoPro events test dataset](https://pan.baidu.com/s/1UKV-sPGo9mRf7XJjZDoF7Q) (code: kmaz) to ./Datasets
- Train the model with default arguments by running

  python main_train.py

## Evaluation
- Download the [GoPro events test dataset](https://pan.baidu.com/s/1UKV-sPGo9mRf7XJjZDoF7Q) (code: kmaz) to ./Datasets
- Download the  [pretrained model](https://pan.baidu.com/s/1qvTokB8mcAA8cj56F1rE4w) (code: daye) to ./checkpoints/models/AHDINet
- Test the model with default arguments by running

  python main_test.py
  
## Citations
    @inproceedings{yang2025asymmetric,
      title={Asymmetric Hierarchical Difference-aware Interaction Network for Event-guided Motion Deblurring},
      author={Yang, Wen and Wu, Jinjian and Li, Leida and Dong, Weisheng and Shi, Guangming},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      volume={39},
      number={9},
      pages={9265--9273},
      year={2025}
    }
  
## Contact
 Should you have any questions, please feel free to contact [wenyang.xd@gmail.com](mailto:wenyang.xd@gmail.com)


## Acknowledgement
Thanks to the inspirations and codes from [MPRNet](https://github.com/swz30/MPRNet)
