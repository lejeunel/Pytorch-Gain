from pascal_voc_loader import PascalVOCLoader
import matplotlib.pyplot as plt
import numpy as np
from transform import DropoutAndAffine, Affine
import data

root = '/home/laurent.lejeune/medical-labeling/VOC2012/'

loader = PascalVOCLoader(root,
                         augmentations=DropoutAndAffine)
                         # augmentations=Affine)

rds = data.RawDataset(
    root,
    num_workers=0,
    output_dims=224,
    augmentations=DropoutAndAffine,
    batch_size_dict={'train': 1, 'test': 1})

# for sample in rds.datasets['train']:
#     print(sample)
sample = loader[10]

ind = 52
sample = loader[ind]

plt.subplot(221)
plt.imshow(sample['image'].transpose((1,2,0)))
plt.subplot(222)
plt.imshow(sample['label'][0][0, ...])
plt.subplot(223)
plt.imshow(sample['label'][1][0, ...])
plt.show()

