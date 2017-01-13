# What's this
Implementation of Deep Networks with Stochastic Depth by chainer  

# Dependencies

    git clone https://github.com/nutszebra/stochastic_depth.git
    cd stochastic_depth
    git submodule init
    git submodule update

# How to run
    python main.py -g 0

# Details about my implementation
All hyperparameters and network architecture are the same as in [[1]][Paper] except for data-augmentation.  
* Data augmentation  
Train: Pictures are randomly resized in the range of [32, 36], then 32x32 patches are extracted randomly and are normalized locally. Horizontal flipping is applied with 0.5 probability.  
Test: Pictures are resized to 32x32, then they are normalized locally. Single image test is used to calculate total accuracy.  

* Drop probability  
As [[1]][Paper] said, P_0 is 1 and P_L is 0.5.


# Cifar10 result
| network                                           | depth  | total accuracy (%) |
|:--------------------------------------------------|--------|-------------------:|
| Deep Networks with Stochastic Depth [[1]][Paper]  | 110    | 94.75              |
| my implementation                                 | 110    | 94.76               |

<img src="https://github.com/nutszebra/stochastic_depth/blob/master/loss.jpg" alt="loss" title="loss">
<img src="https://github.com/nutszebra/stochastic_depth/blob/master/accuracy.jpg" alt="total accuracy" title="total accuracy">

# References  
Deep Networks with Stochastic Depth [[1]][Paper]

[paper]: https://arxiv.org/abs/1603.09382 "Paper"
