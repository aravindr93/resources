# Installing Deep Learning libraries (and resources)

## Tensorflow
Assumes `conda` env has been created and activated (e.g. `source activate tfenv`)

Install `tensorflow` with GPU support enabled. Simplest way is
```
pip install tensorflow-gpu
```
If this fails, we can get it from source and build.
```
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp35-cp35m-linux_x86_64.whl
pip install --ignore-installed --upgrade TF_BINARY_URL
```
Note that the above URL is for the GPU version of tensorflow 1.4 for python 3.5. If any of these things change, find the corresponding binary url here: https://www.tensorflow.org/install/install_linux#InstallingAnaconda
If you get a “locale.Error: unsupported locale setting” during TF installations, enter
```
export LC_ALL=C
```
Then, repeat the installation process.
If no further errors occur, the TF installation is over. However, for GPU acceleration to properly work, we still have to install Cuda Toolkit and cuDNN. See the `gpu` directory in this repository to see how to get these to work.

After installing CUDA and cuDNN, you should be able to test `tensorflow` with the following commands.
```
python
>>> import tensorflow as tf
>>> sess = tf.Session()
```
If you see a message like "Found device 0 with properties..." then you are set. See this link for additional information and debugging tips: https://docs.devicehive.com/v2.0/blog/using-gpus-for-training-tensorflow-models

## Pytorch
