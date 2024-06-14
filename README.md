## Introduction

This package contains scripts for running the Equivar model for prediction of Born effective charges tensor.

## Installation

```sh
pip3 install --upgrade pip
pip install wheel
pip install numpy
```

Install pytorch and pytorch geometric
```sh
pip install torch torchvision torchaudio
```
(if some old version of cuda library is installed, it may be necessary to install specific version of pytorch, see https://pytorch.org/get-started/locally/ for more details)

After downloading `equivar_eval`, install with
`cd equivar_eval ; pip install -r requirements.txt ; pip install . ;`


## Running the script
Run the script using the command `equivar-eval`.
The script requires configuration file, `config.yaml`, containing path to pre-trained model file, dataset dir, and csv output file name. 
The name of the dataset file located in the dataset dir must be `frames.xyz`.
A sample configuration file is given in `examples/configs/config.yaml`
