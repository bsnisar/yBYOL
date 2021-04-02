
# Yet another BYOL implementation


This is yet another practical implementation of an astoundingly simple method for self-supervised learning that achieves 
a new state of the art (surpassing SimCLR) without contrastive learning and having to designate negative pairs.

This repository offers a module can be used to train your own 
net from unlabeled data based on pretrained models.

Inspired by https://github.com/lucidrains/byol-pytorch

![byol](prepare/diagram.png?raw=true)

### Install

```
pip install git+https://github.com/bsnisar/parastash
```

## Usage

```
from parastash import models, datasets, transforms, models_hub

MODEL = 'instagram'
input_shape = 244
output_dimension = 128

model = models.BYOL(
    external_net=models_hub.ExternalModel(name=MODEL),
    input_shape=input_shape,
    output_dimension=output_dimension,
    freeze_base_network=True,
)

train_transformations = transforms.byol_augmentations()
train_dataset = datasets.FolderDataset(
      TRAIN_DIR, transformations=train_transformations
)

loader_workers = min(os.cpu_count() - 2, 0)
epochs = 10

train_loader = train_dataset.create_loader(
    batch_size=64, shuffle=True, drop_last=True, num_workers=loader_workers
)

model.train(
     epochs=epochs,
     loader=train_loader,
     print_metrics=True
)

model.save(SAVE_MODEL_DIR)
```