# TickNets
Tiny Tick-Shape Networks of Full-Residual Point-Depth-Point Blocks for Image Classification

**Abstract:**

* Efficiently rescaling a large dataset by adapting statistical

**Training TickNets on Datasets:**

For Stanford Dogs. Note that it will automatically run for all TickNets, i.e., TickNet-basic, TickNet-small and TickNet-large
```
$ python TickNet_Dogs.py 
```
For ImageNet-1k and Places365: -a large for training TickNet-large; -a small for TickNet-small
```
$ python TickNet_ImageNet.py -a large
$ python TickNet_Places365.py -a large 
```
**Validating the trained models of TickNets**
* For ImageNet-1k and Places365: -a large for training TickNet-large; -a small for TickNet-small.
```
$ python TickNet_ImageNet.py -a large --evaluate
$ python TickNet_Places365.py -a large --evaluate
```
Note: Download the trained model of TickNet-large on Datasets: [For Stanford Dogs](https://drive.google.com/file/d/13K-zgU2nkYCUtEfOQ8awXYnGLYz92_c-/view?usp=drive_link); [For ImageNet-1k](https://drive.google.com/file/d/1-4EusMzLuWzxPQUEKWo0n4iE0H59Qm37/view?usp=drive_link); [For Places365](https://pages.github.com/). And then locate the download file at ./checkpoints/[name_dataset]/large
