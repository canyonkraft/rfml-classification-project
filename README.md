## Files
This repo contains all python scripts and datasets used for the 4622 personal project:
* train_radioml.py
* snr_eval.py
* train_lowsnr.py
* train_highsnr.py
* finetune_panoradio.py

## Space Requirements
You will need 6GB free space:
* Panoradio HF Dataset: 5.3GB
* RML2016.10a Dataset: 600MB
* Model Checkpoints: 4-8MB

## System Dependencies
You will need to install PyTorch to run the relevant code. GPU support is recommended; if you have an NVIDIA GPU, use the following command to install it:
```
# CUDA 12.1
pip install --break-system-packages \
    torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 if you have an older gpu
pip install --break-system-packages \
    torch --index-url https://download.pytorch.org/whl/cu118
```

The core dependencies for this project can be download as such:
```
pip install --break-system-packages \
    torch numpy scipy pandas scikit-learn matplotlib seaborn
```
Or, if you can't install system-wide:
```
pip install --user \
    torch numpy scipy pandas scikit-learn matplotlib seaborn
```

To verify the installation, you can run this code in python:
```
import torch
import numpy as np
import scipy
import pandas as pd
import sklearn
import matplotlib
import seaborn

print(f"PyTorch:      {torch.__version__}")
print(f"CUDA avail:   {torch.cuda.is_available()}")
print(f"NumPy:        {np.__version__}")
print(f"SciPy:        {scipy.__version__}")
print(f"Pandas:       {pd.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"Matplotlib:   {matplotlib.__version__}")
print(f"Seaborn:      {seaborn.__version__}")
```

## Datasets
The best way to download the RadioML2016.10a dataset is to create a free account through [Kaggle](kaggle.com). The link to the dataset can be found [here](kaggle.com/datasets/nolasthitnotomorrow/radioml2016-deepsigcom).

To download the Panoradio HF dataset, you must first acquire the wget package. This can be installed via the following command:
```
sudo apt install wget
```

Then, run these two commands:
```
wget http://www.panoradio-sdr.de/wp-content/uploads/dataset_panoradio_hf.npy
wget http://www.panoradio-sdr.de/wp-content/uploads/dataset_panoradio_hf_tags.csv

# Optional Readme
wget http://www.panoradio-sdr.de/wp-content/uploads/dataset_panoradio_hf_readme.txt
```

## Licenses
The use of the RadioML 2016.10a dataset is for public use under license. It is owned by DeepSig Inc. and is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
This dataset was created and hosted by DeepSig. Their dataset website can be found here:
[Deepsig AI](deepsig.ai/datasets/). This repository and the related project are not used for any commercial purposes. # rfml-classification-project
