import torch

USE_CUDA = torch.cuda.is_available()

DATASET_PATH = './EuroSAT_data'

BATCH_SIZE = 'TODO' # Number of images that are used for calculating gradients at each step

NUM_EPOCHS = 'TODO' # Number of times we will go through all the training images. Do not go over 25

LEARNING_RATE = 'TODO' # Controls the step size

MOMENTUM = 'TODO' # Momentum for the gradient descent

WEIGHT_DECAY = 'TODO' # Regularization factor to reduce overfitting

NUM_CLASSES = 10

LABELS= ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']