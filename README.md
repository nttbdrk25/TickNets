# TickNets
Tiny Tick-Shape Networks of Full-Residual Point-Depth-Point Blocks for Image Classification

**Abstract:**

* Efficiently rescaling a large dataset by adapting statistical

**Training TickNets on Datasets:**

For Stanford Dogs. Note that it will automatically run for all TickNets, i.e., TickNet-basic, TickNet-small and TickNet-large
```
$ python TickNet_Dogs.py 
```
For ImageNet-1k.
```
$ python python TickNet_ImageNet.py -a large 
```
* **Validating the trained models of TickNets**
```
$ python python TickNet_ImageNet.py -a large --evaluate 
```
