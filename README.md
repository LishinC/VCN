# Volume Contrastive Network

- [Overview](#Overview)
- [Run the full workflow](#Run-the-full-workflow)
- [Run the code on your own dataset](#Run-the-code-on-your-own-dataset)
- [Loading the trained model weights](#Loading-the-trained-model-weights)


This is the official repository for Volume Contrastive Network:
```
Li-Hsin Cheng, Xiaowu Sun, Rob J. van der Geest. (2022).
Contrastive learning for echocardiographic view integration."
```

## Overview
![](VCN/Fig1_animate.gif)

In this work, we aimed to tackle the challenge of fusing information from multiple echocardiographic views, mimicking cardiologists making diagnoses with an integrative approach. For this purpose, we used the available information provided in the CAMUS dataset to experiment combining 2D complementary views to derive 3D information of left ventricular (LV) volume. We proposed intra-subject and inter-subject volume contrastive losses with varying margin to encode heterogeneous input views to a shared view-invariant volume-relevant feature space, where feature fusion can be facilitated. The results demonstrated that the proposed contrastive losses successfully improved the integration of complementary information from the input views, achieving significantly better volume predictive performance (MAE: 10.96 ml, RMSE: 14.75 ml, R2: 0.88) than that of the late-fusion baseline without contrastive losses (MAE: 13.17 ml, RMSE: 17.91 ml, R2: 0.83).

## Run the full workflow
First, clone the project and install util_VCN.
```
git clone https://github.com/LishinC/VCN.git
cd VCN/util_VCN
pip install -e .
cd ../VCN
```

Next, download the CAMUS dataset from [the official site](https://www.creatis.insa-lyon.fr/Challenge/camus/participation.html), and put the data inside `data/CAMUS/training`. Specifically, organize the data as following:
```
VCN/
├── VCN
├── util_VCN
└── data
    └── CAMUS
        ├── info.csv
        ├── 5fold
        ├── patient0001
        ├── patient0002
        ├── patient0003
        .
        .
        .
```

Then, to run the full workflow (training, validation, testing), simply run `main.py` under `VCN/VCN`:
```
python main.py
```

## Run the code on your own dataset
The dataloader in `util_VCN` can be modified to fit your own dataset.


## Loading the trained model weights
The model weights are made available for external validation, or as pretraining for other echocardiography-related tasks. Please download the model weights [here](https://drive.google.com/drive/folders/17S3UcjGdXJoUOScE-ydE8_WtXIeRXFNk?usp=sharing) and put the files under `VCN/VCN/model`.

To load the VCN weights, navigate to the `VCN/VCN` folder, and run the following python code:
```
from util_VCN.model.initialize_load_mtview_2Dresnet import initialize_load_model
# from util_VCN.model.initialize_load_sgview_2Dresnet import initialize_load_model # For loading single-view model

model_path = PATH_TO_MODEL_WEIGHTS
model, param = initialize_load_model('train')
model.load_state_dict(torch.load(model_path))
```


## Questions and feedback
For techinical problems or comments about the project, feel free to contact `l.cheng@lumc.nl`.