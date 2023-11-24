# TickNets: Efficient tick-shape networks of full-residual point-depth-point blocks for image classification

**Abstract:**

* Light-weight convolutional neural networks (CNNs) are crucial for deploying computer vision
applications in mobile devices. However, such models ordinarily have the stead-increased channels in their backbone which enlarges the model size and the deficiency of identity mappings in residual connections which leads to modest performance in feature extraction. To mitigate those issues, we propose light-weight networks based on three novel concepts as follows. Firstly, an efficient perceptron is presented to encapsulate point-depth-point (PDP) features extracted by light-weight convolutions along with a full-residual (FR) mechanism through the architecture of a network. This full-residual connection is proposed to deal with the shortcoming of the existing tiny models whose architecture incompletely exploits identity mappings. It is due to the variability of spatial dimension caused by several strides of their convolutional operations. Secondly, a tick-shape backbone is then introduced by designing its structure in accordance with the channel elasticity of the FR-PDP perceptron subject to a shape of check mark. Thirdly, taking advantage of the channel elasticity concept, three tick-shape networks (TickNets) are constructed in a light-weight architecture by hooking one or more tick-shape backbones. Experimental results for image classification on benchmark datasets have clearly corroborated the prominence of the proposed methods.

<u>**Training TickNets on Datasets:**</u>

For Stanford Dogs. Note that it will automatically run for all TickNets, i.e., TickNet-basic, TickNet-small and TickNet-large
```
$ python TickNet_Dogs.py
```
For ImageNet-1k and Places365: -a large for training TickNet-large; -a small for TickNet-small
```
$ python TickNet_ImageNet.py -a small
$ python TickNet_Places365.py -a small 
```
Note: Subject to your system, modify these training files (*.py) to have the right path to datasets

**Validating the trained models of TickNets:**
* For Stanford Dogs (TickNet-small)
```
$ python TickNet_Dogs.py --evaluate
```
* For ImageNet-1k and Places365: -a large for training TickNet-large; -a small for TickNet-small.
```
$ python TickNet_ImageNet.py -a small --evaluate
$ python TickNet_Places365.py -a small --evaluate
```

Note: For instances of validation of TickNet-small, download the trained model of TickNet-small on Datasets: [Click here for Places365](https://drive.google.com/drive/folders/1EdlA3tuOutBJMR23B-fcSOKKB69hAQ5R?usp=sharing); [Click here for ImageNet-1k](https://drive.google.com/drive/folders/1t1M_QJwCmcaTgKBsJBmzrU-kabQeOPDT?usp=sharing); [Click here for Stanford Dogs](https://drive.google.com/drive/folders/1RGglukdrd5xDrGSo6ONmHTCZNZ-YwpZb?usp=sharing). And then locate the downloaded file at ./checkpoints/[name_dataset]/small

**Related citations**
If you use any materials, please cite the following relevant works.

```
@article{neucoTickNetNguyen23,
  author       = {Thanh Tuan Nguyen and Thanh Phuong Nguyen},
  title        = {Efficient tick-shape networks of full-residual point-depth-point blocks for image classification},
  journal      = {Neurocomputing},
  note         = {(submitted in 2023)}
}
```
