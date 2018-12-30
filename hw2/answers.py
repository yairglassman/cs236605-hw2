r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 5, 0.0025117, -9.7
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0.1, 0.001, 0.0001, 0.1, 0
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd, lr = 0.5, 0.00001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Unfortunately, we have different results than the results we thought we will have.
we thought that when we will use the dropout we will have better results.
Maybe due to the randomness of choosing the neurons to turn off, the model is getting really slow 
so we can't see the improve in 30 epochs.
but we think that in this way we get more redundancy to the model so yield can get better - but we need more epochs,
because its so slow.

"""

part2_q2 = r"""
**Your answer:**

increasing of the test loss is possible when training with the cross-entropy loss function.
the reason is the connection between total loss to the accuracy.
the total loss is actually the mean - of the loss func to every one of the classes,
and the accuracy is based on the class that have the top score.
so there is a chance that the top score is for the right class, but for other class that are'nt correct the loss function increase
and than we dill with higher accuracy, and higher total loss. 

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
1. The shallowest network generates the best accuracy in experiment 1.1.
2. The network was not trainable for L=8 and L=16. 
In the first case, we have very deep network (with minimal pooling in comparison to feature expansion). We are not able to train this model with our resources since it has lot of features. 
n the second case, the problem is that we cannot build the model. Indeed, since the initial dimension is 32*32 and we need to divide each of the dimension of H, W by 2 for each MaxPool operation, we obtain a
layer of num_input_feature=0 and therefore we cannot build the classifier.


"""

part3_q2 = r"""
**Your answer:**
Considering the results of experiment 1.2, we can observe that the number of filters K significantly affects the tests accuracy. 
Moreover, we can observe that, as well as in the previous experiment, the shallowest network has better results.


"""

part3_q3 = r"""
**Your answer:**
we didn't succeeded to train in this network.
the problem we got is that the dipper we get, we getting more filters, and it need to be the other way around.
i.e when we get down the layers we use the filters to make features and than we cam reduce the space in the next step.
"""


part3_q4 = r"""
**Your answer:**

1. Since the VGG model worked well on the data set, we decided to add a batchNorm layer after each convolution layer (_make_feature_extractor)
Our classifier block is different and our extractor block is changed in order to maintain the required behavior.
2. The idea is that the values' magnitude remains relatively similar thanks to the normalization layers and this is important for the learning phase. 
This results in an improvement in accuracy.

"""
# ==============
