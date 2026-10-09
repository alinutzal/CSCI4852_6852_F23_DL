# Third-Party Notices

This repository (Apache-2.0, see `LICENSE`) contains course materials for CSCI 4852/6952
Deep Learning, Youngstown State University. Some files are adapted from third-party works.
Their licenses require the notices below to accompany them. Each notice applies to the
adapted portions of the listed files.

Sources were verified against the upstream repositories on 2026-10-09.

## 1. STAT 453, Sebastian Raschka (University of Wisconsin-Madison): MIT License

Sources: <https://github.com/rasbt/stat453-deep-learning-ss20>, <https://github.com/rasbt/stat453-deep-learning-ss21>

- `lab11/1-lenet5-mnist.ipynb`
- `lab11/2-alexnet-cifar10.ipynb`
- `lab11/2-alexnet-cifar10mod.ipynb`
- `lab11/3-cnn-cifar10.ipynb`
- `lab11/4-lenet5-mnist_eval.ipynb`
- `lab11/notes/cross-correlation.ipynb` (from SS21 `L13/code/notes/cross-correlation.ipynb`)
- `lab11/save-and-load/1-lenet5-save.ipynb`
- `lab11/save-and-load/2-lenet5-load.ipynb`
- `lab15/1_VAE_mnist_sigmoid_mse.ipynb`
- `lab15/2_VAE_celeba-sigmoid_mse.ipynb`
- `lab15/3_VAE_nearest-neighbor-upsampling.ipynb`
- `lab15/4_VAE_celeba-inspect-latent.ipynb`
- `lab15/5_VAE_celeba_latent-arithmetic.ipynb`
- `lab2/jupyter-notebook-tutorial.ipynb`
- `lab3/perceptron-animation.ipynb`
- `lab3/perceptron-numpy.ipynb`
- `lab3/perceptron-pytorch_my.ipynb`
- `lab3/vectorization-example.ipynb`
- `lab4/adaline-sgd.ipynb`
- `lab4/adaline-with-autograd.ipynb`
- `lab4/linear-regr-gd.ipynb`
- `lab6/Lab6.ipynb`
- `lab6/cross-entropy-pytorch.ipynb`
- `lab6/logistic-regression.ipynb`
- `lab6/softmax-regression-mnist.ipynb`
- `lab6/softmax-regression_scratch.ipynb`
- `lab7/customDataloader/custom-dataloader-example.ipynb`
- `lab7/customDataloader/train_model.ipynb`
- `lab7/mlp-fromscratch__sigmoid-mse.ipynb`
- `lab7/mlp-pytorch_sigmoid-mse.ipynb`
- `lab7/mlp-pytorch_softmax-crossentr.ipynb`
- `lab7/xor-problem.ipynb`
- `lab8/L2-log-reg.ipynb`
- `lab8/data-augmentation.ipynb`
- `lab8/dropout.ipynb`

```text
MIT License

Copyright (c) 2020 Sebastian Raschka (STAT 453, Spring 2020)
Copyright (c) 2021 Sebastian Raschka (STAT 453, Spring 2021)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. `lab1/cnn-with-pytorch-for-mnist.ipynb`: two sources

- The **MLP part** (architecture, training and per-class test-accuracy code, comments) is adapted from
  Udacity, *deep-learning-v2-pytorch*, `convolutional-neural-networks/mnist-mlp/mnist_mlp_solution.ipynb`
  <https://github.com/udacity/deep-learning-v2-pytorch> (MIT License).
- The **CNN part** (`CNN` class, Adadelta optimizer, `StepLR(gamma=0.7)`) is adapted from the PyTorch
  examples `mnist/main.py` <https://github.com/pytorch/examples> (BSD 3-Clause License).

```text
MIT License

Copyright (c) 2018 Udacity

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

```text
BSD 3-Clause License

Copyright (c) 2017, Pytorch contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## 3. PyTorch Lightning tutorial (Lightning AI): CC BY-SA

The following notebooks are adapted from *Introduction to PyTorch Lightning*
(`lightning_examples/mnist-hello-world`, <https://github.com/Lightning-AI/tutorials>). The
tutorial's metadata declares the license **CC BY-SA**, which is a *share-alike* license: adapted
versions must be released under the same license, not Apache-2.0. **These files are therefore
licensed CC BY-SA 4.0 (<https://creativecommons.org/licenses/by-sa/4.0/>), not under this
repository's `LICENSE`.** Changes were made (MLP model, Weights & Biases logging, OSC paths).

- `lab2/MNIST_pl.ipynb`
- `lab3/1_MNIST_pl.ipynb`
- `lab3/2_MNIST_pl.ipynb`
- `lab3/3_MNIST_pl.ipynb`
- `1_mnist_pl.py`, `2_mnist_pl.py`, `3_mnist_pl.py`, `4_mnist_pl.py` (script versions of the same tutorial)

## 4. `profileTorch.py`: no license found

Adapted from Weights & Biases, *Profile PyTorch Code*
(`colabs/pytorch-lightning/Profile_PyTorch_Code.ipynb`, <https://github.com/wandb/examples>).
That repository declares **no license**, so by default its content is all rights reserved and
redistributing an adapted copy is not clearly permitted. **Recommended:** remove this file from
the public repository, or obtain permission from the authors. The course template
(`dl-course-template`) already records run time and peak GPU memory, which covers its
teaching purpose.

## Original course work

The following files are original work of the course instructor and are covered by this
repository's `LICENSE` (faculty to confirm):

- `lab2/lab2.ipynb`
- `lab4/todoLabExer.ipynb`
