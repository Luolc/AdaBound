# Demos

Here we provide some demos of using AdaBound on several benchmark tasks.
The purpose of these demos is to give an example of how to use it your research, and also
illustrate the rubost performance of AdaBound.

In short, AdaBound can be regarded as an optimizer that dynamically transforms from Adam to SGD as
the training step becomes larger.
In this way, it can **combines the benefits of adaptive methods, viz. fast initial process, and the
good final generalization properties of SGD**.

In most examples, you can observe that AdaBound has a much faster training speed than SGD
in the early stage, and the learning curve is much smoother than that of SGD.
As for the final performance on unseen data, AdaBound can achieve better or similar performance
compared with SGD, and has a considerable improvement over the adaptive methods.

## Demo List
- CIFAR-10 \[[notebook](./cifar10/visualization.ipynb)\] \[[code](./cifar10)\]

## Future Work

We will keep updating the demos in the near future to include more popular benchmarks.
Feel free to leave an issue or send an email to the first author ([Liangchen Luo](mailto:luolc.witty@gmail.com))
if you want to see a specific task which has not been included yet. :D
