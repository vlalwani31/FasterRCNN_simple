import torchvision
from torchvision.models.detection.image_list import ImageList
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from BoxHead import *
from utils import *
from pretrained_models import *
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import statistics
import os

if __name__ == '__main__':

    # Put the path were you save the given pretrained model
    pretrained_path='checkpoint680.pth'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    backbone, rpn = pretrained_models_680(pretrained_path)
    boxHead=BoxHead()

    # we will need the ImageList from torchvision
    from torchvision.models.detection.image_list import ImageList


    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "./data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"

    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    # Standard Dataloaders Initialization
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size

    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    batch_size = 1
    # print("batch size:", batch_size)
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()


    # Here we keep the top 20, but during training you should keep around 200 boxes from the 1000 proposals
    keep_topK=200

    with torch.no_grad():
        for iter, batch in enumerate(test_loader, 0):
            images = batch['images'].to(device)
            labels = batch['labels']
            boxes=batch['bbox']
            # Take the features from the backbone
            backout = backbone(images)

            # The RPN implementation takes as first argument the following image list
            im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
            # Then we pass the image list and the backbone output through the rpn
            rpnout = rpn(im_lis, backout)

            #The final output is
            # A list of proposal tensors: list:len(bz){(keep_topK,4)}
            proposals=[proposal[0:keep_topK,:] for proposal in rpnout[0]]
            # A list of features produces by the backbone's FPN levels: list:len(FPN){(bz,256,H_feat,W_feat)}
            fpn_feat_list= list(backout.values())
            feature_vectors=boxHead.MultiScaleRoiAlign(fpn_feat_list,proposals)
            new_feature_vectors = torch.cat(feature_vectors, dim=0)
            labels_target, regressor_target = boxHead.create_ground_truth(proposals, labels, boxes)
            proposal_target = []
            for bz in range(len(proposals)):
                box = proposals[bz]
                w = torch.abs(box[:,0] - box[:,2])
                h = torch.abs(box[:,1] - box[:,3])
                k = 4 + torch.log2(torch.sqrt(w*h)/224)
                k = torch.clamp(k.to(torch.int), min = 2, max = 5) - 2
                k2 = [[],[],[],[]]
                for i in range(len(k)):
                    k2[k[i]].append(proposals[bz][i,:])
                for i in range(4):
                    if(len(k2[i]) > 0):
                        props = torch.stack(k2[i], dim = 0)
                        proposal_target.append(props)
            proposal_target = torch.cat(proposal_target, dim=0)

            # print("For the proposals We have a list containing "+str(len(proposals))+" tensors")
            # print("Each one with shape "+str(proposals[0].shape))
            # print("")
            # print("For the features we have a list of features for each FPN level with shapes")
            # for feat in fpn_feat_list:
            #     print(feat.shape)


            # Visualization of the proposals
            for i in range(batch_size):
                img_squeeze = transforms.functional.normalize(images[i,:,:,:].to('cpu'),
                                                              [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                              [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
                fig,ax=plt.subplots(1,1)
                ax.imshow(img_squeeze.permute(1,2,0))


                # for box in proposals[i]:
                #     box=box.view(-1)
                #     rect=patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,color='b')
                #     ax.add_patch(rect)
                # plt.show()
                for j in range(labels_target.shape[0]):
                    if (labels_target[j] != 0):
                        box = proposal_target[j,:]
                        rect=patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,color='b')
                        ax.add_patch(rect)
                for j in boxes[i]:
                    rect=patches.Rectangle((j[0],j[1]),j[2]-j[0],j[3]-j[1],fill=False,color='r')
                    ax.add_patch(rect)
                plt.savefig('./no_background_200_props' + str(iter) + '.png')

            if iter>=5:
                break
