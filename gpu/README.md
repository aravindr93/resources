# Setting up GPU

- Check if your machine has a GPU! If there is a window, peak inside! Otherwise try
```
$ lspci
```
and look for a line that says something like `VGA compatible controller: ...`
- Once we know that the machine has a GPU, we first want to install CUDA. Go to https://developer.nvidia.com/cuda-downloads and download the appropriate version. Download the `deb (local)` version. Follow the instructions from the link to complete the installation.
- After installing cuda, we need to install cudnn. Download the appropriate version from https://developer.nvidia.com/cudnn. You may have to register and get a login. Download the version with the `tgz` extension and *not* the `.deb` file. 
```
sudo tar -xvf <cudnn.....>.tgz -C /usr/local
```
- Add following lines to `bashrc`:
```
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
```
