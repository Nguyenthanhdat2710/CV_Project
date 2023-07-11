from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix  
import numpy as np

def reformat_img(org):
  palette = {(0,0,0): 0,  (128,0,0): 1,  (0,128,0): 2,  (128,128,0): 3,  (0,0,128): 4,  (128,0,128): 5,  (0,128,128): 6,  (128,128,128): 7,  
					   (64,0,0): 8,  (192,0,0): 9,  (64,128,0): 10,  (192,128,0): 11,  (64,0,128): 12,  (192,0,128): 13,  (64,128,128): 14,  (192,128,128): 15,  
					   (0,64,0): 16,  (128,64,0): 17,  (0,192,0): 18,  (128,192,0): 19,  (0,64,128): 20}
  if len(org.shape) > 2:
    data = np.zeros([org.shape[0], org.shape[1]], dtype=np.uint8)
    for i in range(org.shape[0]):
      for j in range(org.shape[1]):
        data[i][j] = palette[tuple(org[i][j][::-1])]
    return data
  else:
    return np.where(org == 255, 0, org)

gt_dir = 'VOCdevkit/VOC2012/SegmentationClassAug'
pd_dir = 'data/proxy_label'
img_path = 'VOCdevkit/VOC2012/ImageSets/Segmentation/train_cls.txt'
img_list = []
with open(img_path, 'r') as f:
  for line in f:
    img_list.append(line.split()[0])

from sklearn.metrics import jaccard_score
scores = np.zeros(len(img_list))
for idx, img in enumerate(img_list):
  gt_img = gt_dir + '/' + img + '.png'
  pd_img = pd_dir + '/' + img + '.png'
  gt = reformat_img(cv2.imread(gt_img, 0)).flatten()
  pd = reformat_img(cv2.imread(pd_img, 1)).flatten()
  scores[idx] = jaccard_score(gt, pd, average='macro')
  print(f'Progress: {idx + 1}/{len(img_list)} Mean: {scores[:idx + 1].mean()}')

