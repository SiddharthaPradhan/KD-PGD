# Responsible AI Final Project

Group 1

## Advesarial Sample Generation with Knowledge Distillation

Instructions for setup:

1. run `get_data.py` to download and setup CIFAR-10.
2. Download pretrained weights for teachers and blackbox from [here](https://drive.usercontent.google.com/download?id=17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq&export=download&authuser=0). Extract zip inside `models` folder.
3. run `test_models.py` to check if the weights have been loaded.

Instructions for training:

1. run `train_student.py` with selected args to train the student

### Model Metrics

-   ResNet50 (Teacher 1): Test=91.39%
-   DenseNet161 (Teacher 2): Test=92.21%
-   GoogLeNet (BlackBox Model): Test=91.26%
-   ResNet18 (Pretrained): Test=90.1%
-   ResNet18 (Sid Trained): Test=87.27%

### Student Model Metrics

-   Student V1 (Simple Teacher Switching 4 epochs, a=0, t=1) : Test=90.31%
-   Student V2 (Joint Training, a=0, t=1) : Test=90.82%
-   Student V2 (Joint Training, a=3, t=1) : Test=91.02%
