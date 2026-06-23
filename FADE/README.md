
[FADE: Few-shot/zero-shot Anomaly Detection Engine using Large Vision-Language Model](https://arxiv.org/abs/2409.00556)
---
This repository contains the edited implementation of 
[FADE: Few-shot/zero-shot Anomaly Detection Engine using Large 
Vision-Language Model](https://arxiv.org/abs/2409.00556), BMVC 2024.

## Prerequisites

For dependencies see `poetry.yaml`.

### Install python dependencies
```
poetry install
```
### startup notebook
```
.\venv\Scripts\Activate
poetry run jupyter lab
```

### how to run on data
First put the hdf5 file of the waver in the data folder. Then in the notebook change the line 
```
IMAGE_PATH = "data/..." 
```
in the fourth code cell.

### how to run with Few shot
In the folder Few-shot-examples, add your clean data. To activate the few-shot method set 
```
FEW_SHOT = False # Set to true to activate few-shot. 
```
to true in the second cell.