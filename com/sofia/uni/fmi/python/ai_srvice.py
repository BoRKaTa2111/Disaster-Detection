from torchvision import datasets

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

DATA_PATH = "data/Comprehensive_Disaster_Dataset(CDD)"
data_sets = datasets.ImageFolder(DATA_PATH)

print("Classes:", data_sets.classes)
print("Total images:", len(data_sets))
