import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        #############################################
        # TODO Initialize  Dataset
        imgs_path, mask_path, labels_path, bounding_box_path = path
        hf = h5py.File(imgs_path, 'r')
        self.imgs_data = hf.get('data')
        self.labels_data = np.load(labels_path, allow_pickle=True, encoding='latin1')
        hf = h5py.File(mask_path, 'r')
        self.mask_data = hf.get('data')
        self.bounding_box_data = np.load(bounding_box_path, allow_pickle=True, encoding='latin1')
        self.p1 = transforms.Compose([transforms.Resize((800,1066)), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), transforms.Pad((11,0))])
        self.p2 = transforms.Compose([transforms.Resize((800,1066)), transforms.Pad((11,0))])
        #############################################


    # In this function for given index we rescale the image and the corresponding  masks, boxes
    # and we return them as output
    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
        # index
    def __getitem__(self, index):
        ################################
        # TODO return transformed images,labels,masks,boxes,index
        img = torch.tensor(self.imgs_data[index]/255, dtype = torch.float)
        label = torch.tensor(self.labels_data[index], dtype = torch.float)
        offset = 0
        for i in range(index):
          offset += len(self.labels_data[i])
        mask = torch.tensor(self.mask_data[offset:(offset + len(self.labels_data[index])), :, :] / 1.0, dtype = torch.float)
        bbox = torch.tensor(self.bounding_box_data[index], dtype = torch.float)
        transed_img, transed_mask, transed_bbox = self.pre_process_batch(img, mask, bbox)
        ################################
        assert transed_img.shape == (3,800,1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]

        
        return transed_img, label, transed_mask, transed_bbox, index



    # This function preprocess the given image, mask, box by rescaling them appropriately
    # output:
    #        img: (3,800,1088)
    #        mask: (n_box,800,1088)
    #        box: (n_box,4)
    def pre_process_batch(self, img, mask, bbox):
        #######################################
        # TODO apply the correct transformation to the images,masks,boxes
        img = self.p1(img.unsqueeze(0))
        mask = self.p2(mask.unsqueeze(0))
        multip = torch.ones_like(bbox)
        multip[:,0] = 2.665 # Y1 scaling
        multip[:,1] = 8/3   # X1 scaling
        multip[:,2] = 2.665 # Y2 scaling
        multip[:,3] = 8/3   # X3 scaling
        bbox = (bbox * multip).int()
        multip = torch.zeros_like(bbox)
        multip[:,0] = 11  # Y1 padding
        multip[:,1] = 0 # X1 padding
        multip[:,2] = 11  # Y2 padding
        multip[:,3] = 0 # X2 padding
        bbox = bbox + multip
        ######################################

        assert img.squeeze(0).shape == (3, 800, 1088)
        assert bbox.shape[0] == mask.squeeze(0).shape[0]

        return img.squeeze(0), mask.squeeze(0), bbox
    

    
    def __len__(self):
        return len(self.imgs_data)




class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers


    # output:
    #  dict{images: (bz, 3, 800, 1088)
    #       labels: list:len(bz)
    #       masks: list:len(bz){(n_obj, 800,1088)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #       index: list:len(bz)
    def collect_fn(self, batch):
        images = []
        labels = []
        masks = []
        bounding_boxes = []
        indexes = []
        for image, label, mask, bbox, index in batch:
            images.append(image)
            labels.append(label)
            masks.append(mask)
            bounding_boxes.append(bbox)
            indexes.append(index)
        out_batch = {"images": torch.stack(images, dim = 0), "labels": labels, "masks": masks, "bbox": bounding_boxes, "index": indexes}
        return out_batch


    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)
def visualize(image,label,mask,bounding_box, index):
  img = torch.clamp(image,min=0,max=1)
  e,r,c = img.shape
  for j in range(len(label)):
    # Make Bounding Box
    if (len(bounding_box) != 0):
      y_1 = int(bounding_box[0][j][0])
      x_1 = int(bounding_box[0][j][1])
      y_2 = int(bounding_box[0][j][2])
      x_2 = int(bounding_box[0][j][3])

      if x_1 < 0:
        x_1 = 0
      if x_1 >= r:
        x_1 = r-1

      if x_2 < 0:
        x_2 = 0
      if x_2 >= r:
        x_2 = r-1

      if y_1 < 0:
        y_1 = 0
      if y_1 >= c:
        y_1 = c-1

      if y_2 < 0:
        y_2 = 0
      if y_2 >= c:
        y_2 = c-1

      #Red image
      img[0,x_1:x_2,y_1] = 1
      img[0,x_1, y_1:y_2] = 1
      img[0,x_2, y_1:y_2] = 1
      img[0,x_1:x_2, y_2] = 1

      #Zeroing Green image
      img[1,x_1:x_2,y_1] = 0
      img[1,x_1, y_1:y_2] = 0
      img[1,x_2, y_1:y_2] = 0
      img[1,x_1:x_2, y_2] = 0

      #Zeroing Blue image
      img[2,x_1:x_2,y_1] = 0
      img[2,x_1, y_1:y_2] = 0
      img[2,x_2, y_1:y_2] = 0
      img[2,x_1:x_2, y_2] = 0
    # Make Mask
    for i in range(r):
      for k in range(c):
        if (mask[0][j,i,k] != 0):
          if (label[0][j] == 1):
            img[0,i,k] = 0.25 + (0.75 * img[0,i,k])
            img[1,i,k] = 0
            img[2,i,k] = 0
          elif (label[0][j] == 2):
            img[0,i,k] = 0
            img[1,i,k] = 0.25 + (0.75 * img[1,i,k])
            img[2,i,k] = 0
          else:
            img[0,i,k] = 0
            img[1,i,k] = 0
            img[2,i,k] = 0.25 + (0.75 * img[2,i,k])
  plt.figure(figsize=(7,7))
  plt.imshow(torch.moveaxis(img, 0, -1))
  plt.savefig('./vis' + str(index[0]) + '.png')

def plot_fn(data,title,x_label,y_label, nam):

  plt.figure(figsize=(7,7))
  plt.plot(range(len(data)),data)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(title)
  plt.savefig('./' + nam + '.png')


if __name__ == '__main__':
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)
    
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    #rpn_net = RPNHead()
    
    batch_size = 4
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()
