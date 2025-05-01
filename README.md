# Responsible AI Final Project

Group 1

## Advesarial Sample Generation with Knowledge Distillation

Instructions for setup:

1. run `get_data.py` to download and setup CIFAR-10.
2. Download pretrained weights for teachers and blackbox from [here](https://drive.usercontent.google.com/download?id=17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq&export=download&authuser=0). Extract zip inside `models` folder.
3. run `test_models.py` to check if the weights have been loaded.

Instructions for training:

-   run `train_{"type"}.py` with selected args to train the student with type={"type"}

The trained weights for the students models are provided [here](https://drive.google.com/drive/folders/1PEUiJuVprx271w_Pno4J-onHww0CxH1A?usp=sharing).
Note. Experiment 1 uses lower max lr = 1e-3, and Experiment 2 uses higher max lr = 1e-2. The main results presented in the paper are from Experiment 2.

Checkpoints are saved as 'stu_resnet18\_{'type'}\_a{'alpha'}\_t\_{'tau'}.cpt'. Types = {""} for Curricula based student (Type 1) and {"multiple\*"} for student trained jointly with multiple teachers.

### Model Metrics

-   ResNet50 (Teacher 1): Test=91.39%
-   DenseNet161 (Teacher 2): Test=92.21%
-   GoogLeNet (BlackBox Model): Test=91.26%
-   ResNet18 (Pretrained): Test=90.1%
-   ResNet18 (Sid Trained): Test=87.27%
