##################
# ANN VGG16: 92.18
##################
### --thresh 1 since `ckpts/CIFAR10/vgg16/vgg16-cifar10-relu[1].pth`
#--------------------------------------
### 92.39 %, paper: 92.29 %
# DATASET=CIFAR10
# ARCH=VGG16
# T=16
# SHIFT_SNN=$T
# THRESH=1
#--------------------------------------
### 92.26 %, paper: 92.29 %
# DATASET=CIFAR10
# ARCH=VGG16
# T=32
# SHIFT_SNN=$T
# THRESH=1
#--------------------------------------
### 92.20 % / 92.23 %, paper: 92.22 %
# DATASET=CIFAR10
# ARCH=VGG16
# T=64
# SHIFT_SNN=$T
# THRESH=1
#--------------------------------------
### 92.25 %, paper: 92.24 %
# DATASET=CIFAR10
# ARCH=VGG16
# T=128
# SHIFT_SNN=$T
# THRESH=1
#--------------------------------------

#==============================================================================#

#######################
# ResNet20 ann: 93.25 %
#######################

#--------------------------------------
### 91.53 %, paper: 92.41 %
# DATASET=CIFAR10
# ARCH=ResNet20
# T=16
# SHIFT_SNN=$T # 16 50+%
# THRESH=1
#--------------------------------------
### 92.42 %; paper: 93.30 %
# DATASET=CIFAR10
# ARCH=ResNet20
# T=32
# SHIFT_SNN=$T
# THRESH=1
#--------------------------------------
### ann: 92.87 %; paper: 93.55 %
# DATASET=CIFAR10
# ARCH=ResNet20
# T=64
# SHIFT_SNN=$T
# THRESH=1
#--------------------------------------
### 93.09 %; paper: 93.55 %
# DATASET=CIFAR10
# ARCH=ResNet20
# T=128
# SHIFT_SNN=$T
# THRESH=1
#--------------------------------------

#==============================================================================#

#############################
# ANN VGG16 cifar100: 70.38 %
#############################

#--------------------------------------
# 62.63 % ; paper 65.94 %
# DATASET=CIFAR100
# ARCH=VGG16
# T=16
# SHIFT_SNN=$T
# THRESH=2
#--------------------------------------

#--------------------------------------
### 70.14 % ; paper: 69.80
# DATASET=CIFAR100
# ARCH=VGG16
# T=32
# SHIFT_SNN=$T
# THRESH=2
#--------------------------------------

#--------------------------------------
### 70.37 % ; paper: 70.35 %
# DATASET=CIFAR100
# ARCH=VGG16
# T=64
# SHIFT_SNN=$T
# THRESH=2
#--------------------------------------

#--------------------------------------
### 70.57 % ; paper: 70.47 %
# DATASET=CIFAR100
# ARCH=VGG16
# T=128
# SHIFT_SNN=$T
# THRESH=2
#--------------------------------------

#==============================================================================#

####################################
### ANN, ResNet20, CIFAR100, 69.80 %
####################################

#--------------------------------------
### 62.50 % ; paper: 63.73 %
# DATASET=CIFAR100
# ARCH=ResNet20
# T=16
# SHIFT_SNN=$T
# THRESH=2
#--------------------------------------

#--------------------------------------
### 68.14 % ; paper: 68.40 %
# DATASET=CIFAR100
# ARCH=ResNet20
# T=32
# SHIFT_SNN=$T
# THRESH=2
#--------------------------------------

#--------------------------------------
### 69.31 % ; paper: 69.27 %
# DATASET=CIFAR100
# ARCH=ResNet20
# T=64
# SHIFT_SNN=$T
# THRESH=2
#--------------------------------------

#--------------------------------------
### 69.61 % ; paper: 69.49 %
# DATASET=CIFAR100
# ARCH=ResNet20
# T=128
# SHIFT_SNN=$T
# THRESH=2
#--------------------------------------

python main_simulation.py \
    --dataset $DATASET \
    --batch_size 128 \
    --arch $ARCH \
    --thresh $THRESH \
    --T $T \
    --shift_snn $SHIFT_SNN
