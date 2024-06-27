## Introduction

This package contains scripts for running the Equivar model for prediction of Born effective charges tensor. In particular, BM1 and BM2 models can be used. 

## Installation

```sh
pip3 install --upgrade pip
pip install wheel
pip install numpy
```

Install pytorch
```sh
pip install torch
```
(if some old version of cuda library is installed, it may be necessary to install specific version of pytorch, see https://pytorch.org/get-started/locally/ for more details)

After downloading `equivar_eval`, install with
`cd equivar_eval ; pip install -r requirements.txt ; pip install . ;`


## Running the script
Run the script using the command `equivar-eval`.
The script requires configuration file, `config.yaml`, containing path to pre-trained model file, dataset dir, and csv output file name. 
The BM1 and BM2 pretrained models can be downloaded from https://doi.org/10.17632/hx8kcpxh84.1
The name of the structure file located in the dataset dir must be `frames.xyz`.
A sample configuration file is given in `examples/configs/config.yaml`.
The outputs are tensors of atomic Born effective charges, ($Z_{11}$, $Z_{12}$, $\ldots$, $Z_{33}$), one atom per line, in the csv format.
