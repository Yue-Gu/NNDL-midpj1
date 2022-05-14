from datetime import datetime

# mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# mean and std of cifar10 dataset
CIFAR10_TRAIN_MEAN = (0.4940607, 0.4850613, 0.45037037)
CIFAR10_TRAIN_STD = (0.20085774, 0.19870903, 0.20153421)

# directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

# total training epoches
EPOCH = 100
MILESTONES = [40,80]

# initial learning rate
# INIT_LR = 0.1

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
# time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

# tensorboard log dir
LOG_DIR = 'runs'

# save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 100
