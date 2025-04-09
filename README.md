# Responsible AI Final Project

Group 1

## Advesarial Sample Generation with Knowledge Distillation

Instructions for setup:

1. run `get_data.py` to download and setup CIFAR-10.
2. Download pretrained weights for teachers and blackbox from [here](https://drive.usercontent.google.com/download?id=17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq&export=download&authuser=0). Extract zip inside `models` folder.
3. run `teachers.py` to check if the weights have been loaded.

Instructions for training:

1. run `train_student.py` with selected args to train the student

### Model Metrics

-   ResNet50 (Teacher): Train=94.3%, Val=90.9%
-   DenseNet (Teacher): Train=95.5%, Val=91.8%
-   GoogLenet (BlackBox Model): Train=96.6%, Val=90.9%
-   ResNet18 (Not Student): Train=96.0%, Val=91.2%
