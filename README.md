# Unleashing the Potential of CNNs for Interpretable Few-Shot Learning
This repo contains codes for the paper on both [arXiv](https://arxiv.org/abs/1711.08277) and [OpenReview](https://openreview.net/forum?id=BJ_QxP1AZ). Currently, this repo is at an *very early* version since we just adopted it from an internal experimental repo. We will keep working on developing this repo.

## Dependencies
The following libaries are necessary:   
[PyTorch](http://pytorch.org/) >= 0.3.0   
[NumPy](http://www.numpy.org/) >= 1.13.3   
[SciPy](https://www.scipy.org/) >= 1.0.0   
 
## Training   
* Data    
We use Mini-ImageNet as our benchmark. So at first, you need to download images from [ImageNet](http://www.image-net.org/) according to the splits in the `miniimagenet-csv` directory. Please note that all the splits are adopted from [Ravi & Larochelle](https://openreview.net/forum?id=rJY0-Kcll).

* Pre-trained CNN   
This repo is for the few-shot learning based on a pre-trained ordinary CNN. Thus, to carry out few-shot training you need to prepare a CNN which is trained on the training split of Mini-ImageNet.

* Hyper Settings   
Basically, we use `argparse` to manage the hyper settings. You can see all the options by `python main.py -h`.

* Models   
As our paper presented, there are two models for few-shot learning. One is *Nearest Neighbor on VC-Encodings* (denoted as `NN` in this repo). The other is *Factorizable Likelihood Model* (denoted as `Likelihood` in this repo). You can choose either of these two by setting the `-model` option.

* Sample running script   
Though you can `python main.py --with_all_the_settings_you_want`,  we also provide a sample running script (please see the `run.sh`). You can first change all the `Path_to_*` options to your paths. Then, just run it!

## Contact    
For any questions about this repo or the paper, please feel free to drop a mail to Boyang ([billydeng96@gmail.com](mailto:billydeng96@gmail.com)).
