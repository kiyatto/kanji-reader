# converts ETL9G images (after processing from binary format to png) and labels into Numpy arrays

import os
import numpy as np
from PIL import Image

# points to the folder containing folders of images
data_path = '../etlcdb-image-extractor/etl_data/images/ETL9G'

class_list = sorted([f for f in os.listdir(data_path) 
                      if os.path.isdir(os.path.join(data_path, f))])
class_dict = { name: i for i, name in enumerate(class_list) }

print("Found {} classes: {}".format(len(class_list), class_list))

images = []
labels = []

print("Extracting images and labels from folders...")

for folder_name in class_list:
    folder_path = os.path.join(data_path, folder_name)
    idx = class_dict[folder_name]
    for file in os.listdir(folder_path):
        if not file.endswith('.png'):
            continue
        img = Image.open(os.path.join(folder_path, file)).convert('L') # grayscale
        images.append(np.asarray(img))
        labels.append(idx)

X = np.stack(images);
Y = np.array(labels);

print("Now saving the data to kanji_images.npy, kanji_labels.npy, and class_list.npy...")

np.save("kanji_images.npy", X)
np.save("kanji_labels.npy", Y)
np.save("class_list.npy", np.array(class_list))

