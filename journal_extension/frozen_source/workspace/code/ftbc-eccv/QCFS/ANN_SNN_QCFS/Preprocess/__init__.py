from .getdataloader import *

def datapool(DATANAME, batchsize=8, dist_sample=False):
    if DATANAME.lower() == 'cifar10':
        return GetCifar10(batchsize)
    elif DATANAME.lower() == 'cifar100':
        return GetCifar100(batchsize)
    elif DATANAME.lower() == 'imagenet':
        return GetImageNet(train_batch_size=batchsize, val_batch_size=batchsize, dist_sample=dist_sample)
    else:
        print("still not support this model")
        exit(0)