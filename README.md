# TickNets
Tiny Tick-Shape Networks of Full-Residual Point-Depth-Point Blocks for Image Classification

**Abstract:**

* Efficiently rescaling a large dataset by adapting statistical

**<u>Training TickNets on Datasets:</u>**

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
* For Stanford Dogs (TickNet-large)
```
$ python TickNet_Dogs.py --evaluate
```
* For ImageNet-1k and Places365: -a large for training TickNet-large; -a small for TickNet-small.
```
$ python TickNet_ImageNet.py -a large --evaluate
$ python TickNet_Places365.py -a large --evaluate
```
Note: Download the trained model of TickNet-large on Datasets: [Click here for Places365](https://drive.google.com/file/d/13K-zgU2nkYCUtEfOQ8awXYnGLYz92_c-/view?usp=drive_link); [Click here for ImageNet-1k](https://drive.google.com/file/d/1-4EusMzLuWzxPQUEKWo0n4iE0H59Qm37/view?usp=drive_link); [Click here for Stanford Dogs](https://drive.google.com/file/d/1hb4YCqNeyho03qzPAeFsIp1esxJuef_u/view?usp=drive_link). And then locate the downloaded file at ./checkpoints/[name_dataset]/large
