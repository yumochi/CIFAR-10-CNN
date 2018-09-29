# CIFAR-10-CNN

Just a simple 8 layer CNN developed to work on CIFAR-10. The code is inspired by pytorch mnist tutorial, ie https://github.com/pytorch/examples/blob/master/mnist/main.py as well as work from CS598 D from UIUC, which is a 
Deep Learning course I am taking.

Just a fun little project to show case some basic concepts like max pooling, batch normalization, and drop out.

## Getting Started

To get started on the project is very easy, just

```
git clone git@github.com:yumochi/CIFAR-10-CNN.git
```

### Prerequisites

To run the code you will need the following:

# Python

(Refer to https://www.python.org/downloads/)

# Pytorch 

```
# Python 3.x
pip3 install torch torchvision
```

```
# Python 2.x`
pip install torch torchvision
```

(Refer to https://pytorch.org/get-started/locally/ for more info.)



# Torchvision

```
pip install torchvision
```

(Refer to https://pypi.org/project/torchvision/0.1.8/ )

# h5py (Not necessary)

```
pip install h5py
```

h5py was originally used to import image data, but the code is adopted to use Torchvision

Comment out code if not needed

## Running the tests

# To run the code, just run

```
python main.py
```
# Set parameters with argparser

For a list of terminal commands for the argparser, refer to texts below or check in hw3.py for all parameters

# Set batch-size with --batch-size x

x has to be an integer

```
python hw3.py --batch-size 16
```

# Set epoch number with --epochs x

x has to be an integer

```
python hw3.py --epochs 30
```

# Set learning rate with --lr x

x has to be an float

```
python hw3.py --lr 0.0001
```

# Set sample number in monte carlo approximation with --mck x

x has to be an integer

```
python hw3.py --mck 16
```

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Yumo Chi** - *Initial work* - [CIFAR-10-CNN](https://github.com/yumochi/CIFAR-10-CNN)


## Acknowledgments

* Pytorch Developers
* UIUC CS598D's professor and tas.
