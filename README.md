## Introduction

This package contains scripts for running the Equivar model for prediction of Born effective charges tensor. In particular, BM1 and BM2 models can be used. 

## Installation

After downloading `equivar_eval`, install with
```sh
pip3 install --upgrade pip
pip install wheel
cd equivar_eval ; pip install -r requirements.txt ; pip install . ;
```


Installations of `pytorch` and `torch_geometric` should proceed automatically from `requirements.txt` without any further actions required from user. If this is not the case, see https://pytorch.org/get-started/locally/ for `pytorch` installation, and https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html for `torch_geometric` installation.


## Running the script
Run the script using the command `equivar-eval`.
The script requires configuration file, `config.yaml`, containing path to pre-trained model file, dataset dir, and csv output file name. 
The BM1 and BM2 pretrained models can be downloaded from https://doi.org/10.17632/hx8kcpxh84.1
The name of the structure file located in the dataset dir must be `frames.xyz`.
A sample configuration file is given in `examples/configs/config.yaml`.
The outputs are tensors of atomic Born effective charges, ($Z_{11}$, $Z_{12}$, $\ldots$, $Z_{33}$), one atom per line, in the csv format.
