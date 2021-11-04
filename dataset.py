import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from rpn import *
import matplotlib.patches as patches
import seaborn as sb
import statistics


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
    # input: Int
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

    def get_bbox(self, index):
        bbox = torch.tensor(self.bounding_box_data[index], dtype = torch.float)
        multip = torch.ones_like(bbox)
        multip[:,0] = 8/3   # Y1 scaling
        multip[:,1] = 2.665 # X1 scaling
        multip[:,2] = 8/3   # Y2 scaling
        multip[:,3] = 2.665 # X3 scaling
        bbox = (bbox * multip).int()
        multip = torch.zeros_like(bbox)
        multip[:,0] = 0  # Y1 padding
        multip[:,1] = 11 # X1 padding
        multip[:,2] = 0  # Y2 padding
        multip[:,3] = 11 # X2 padding
        bbox = bbox + multip
        return bbox


    # This function preprocess the given image, mask, box by rescaling them appropriately
    # input:
    #        img: (3, 300, 400)
    #        mask: (n_box, 800, 1088)
    #        box: (n_box, 4)
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

if __name__ == '__main__':
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)
    # Get Histogram
    # asp_hist = []
    # sca_hist = []
    # for i in range(len(dataset)):
    #     b = dataset.get_bbox(i)
    #     # print(b)
    #     w = b[:,3] - b[:,1]
    #     h = b[:,2] - b[:,0]
    #     asp = w / h
    #     sca = torch.sqrt(w * h)
    #     for j in range(b.size(dim=0)):
    #         asp_hist.append(asp[j].item())
    #         sca_hist.append(sca[j].item())
    # print("Aspect Ratio: ", statistics.median(asp_hist))
    # plt.figure(figsize=(7,7))
    # plt.hist(asp_hist, bins=30)
    # plt.xlabel('Aspect Ratio')
    # plt.ylabel('Frequency')
    # plt.title('Aspect Ratio Histogram')
    # plt.savefig('./histogram_Aspect.png')
    # print("Scale: ", statistics.median(sca_hist))
    # plt.figure(figsize=(7,7))
    # plt.hist(sca_hist, bins=30)
    # plt.xlabel('Scale')
    # plt.ylabel('Frequency')
    # plt.title('Scale Histogram')
    # plt.savefig('./histogram_Scale.png')


    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    rpn_net = RPNHead()
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 1
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()
    index_set = (56, 69, 572, 3081)
    for i,batch in enumerate(index_set,0):
        ba = train_build_loader.collect_fn([dataset[batch]])
        images=ba['images'][0,:,:,:]
        indexes=ba['index']
        boxes=ba['bbox']
        masks = ba['masks']
        labels = ba['labels']
        images = transforms.functional.normalize(images, [-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225], inplace=False)
        visualize(images,labels,masks, boxes, indexes)

    for i,batch in enumerate(train_loader,0):
    # for i,batch in enumerate(index_set,0):
        if(i>4):
            break
        images=batch['images'][0,:,:,:]
        indexes=batch['index']
        boxes=batch['bbox']
        # ba = train_build_loader.collect_fn([dataset[batch]])
        # images=ba['images'][0,:,:,:]
        # indexes=ba['index']
        # boxes=ba['bbox']
        gt,ground_coord=rpn_net.create_batch_truth(boxes,indexes,images.shape[-2:])
        # print("Indexes: ", indexes)
        # heat_map = sb.heatmap(gt.squeeze(0).squeeze(0))
        # plt.xlabel("Grid Column")
        # plt.ylabel("Grid Row")
        # plt.title("Heatmap")
        # plt.show()


        # Flatten the ground truth and the anchors
        flatten_coord,flatten_gt,flatten_anchors=output_flattening(ground_coord,gt,rpn_net.get_anchors())

        # Decode the ground truth box to get the upper left and lower right corners of the ground truth boxes
        decoded_coord=output_decoding(flatten_coord,flatten_anchors)

        # Plot the image and the anchor boxes with the positive labels and their corresponding ground truth box
        images = transforms.functional.normalize(images,
                                                      [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                                      [1/0.229, 1/0.224, 1/0.225], inplace=False)
        fig,ax=plt.subplots(1,1)
        ax.imshow(images.permute(1,2,0))

        find_cor=(flatten_gt==1).nonzero()
        find_neg=(flatten_gt==0).nonzero()

        for elem in find_cor:
            coord=decoded_coord[elem,:].view(-1)
            anchor=flatten_anchors[elem,:].view(-1)

            col='r'
            # print("coordinates: ", coord[0], coord[1], coord[2], coord[3])
            rect=patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color=col)
            ax.add_patch(rect)
            rect=patches.Rectangle((anchor[1]-anchor[3]/2,anchor[0]-anchor[2]/2),anchor[3],anchor[2],fill=False,color='b')
            ax.add_patch(rect)

        plt.savefig('./gt' + str(i) + '.png')
