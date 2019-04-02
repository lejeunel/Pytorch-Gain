from pascal_voc_loader import PascalVOCLoader
import matplotlib.pyplot as plt
import numpy as np
import transform
import data

root = '/home/laurent.lejeune/medical-labeling/VOC2012/'

aug_ = getattr(transform, 'DropoutAndAffine')
loader = PascalVOCLoader(root,
                         augmentations=aug_())
                         # augmentations=Affine)

rds = data.RawDataset(
    root,
    num_workers=0,
    output_dims=224,
    augmentations=aug_(),
    batch_size_dict={'train': 1, 'test': 1})

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
