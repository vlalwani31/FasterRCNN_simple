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

def plot_fn(data,title,x_label,y_label, nam):

  plt.figure(figsize=(7,7))
  plt.plot(range(len(data)),data)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(title)
  plt.savefig('./' + nam + '.png')

if __name__ == '__main__':
    # Put the path were you save the given pretrained model
    pretrained_path='checkpoint680.pth'
    device = torch.device('cpu')
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    backbone, rpn = pretrained_models_680(pretrained_path)
    # backbone = backbone.to(device)
    # rpn = rpn.to(device)
    boxHead=BoxHead()
    boxHead=boxHead.to(device)
    keep_topK=200
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    batch_size = 1
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()
    # Code for Training
    training_loss_total = []
    training_loss_r = []
    training_loss_c = []
    validation_loss_total = []
    validation_loss_r = []
    validation_loss_c = []
    model_path = 'model.pth'
    final_items = [i for i in os.listdir('./') if 'model' in i]
    if (model_path in final_items):
        checkpoint = torch.load(train_model_path)
        boxHead.load_state_dict(checkpoint['box_head_state_dict'])
    else:
        optimizer = torch.optim.Adam(boxHead.parameters())
        epochs = 10
        for i in range(epochs):
            # Training Time
            loss_t_list = []
            loss_c_list = []
            loss_r_list = []
            for j,batch in enumerate(train_loader,0):
                images=batch['images']
                indexes=batch['index']
                boxes=batch['bbox']
                masks = batch['masks']
                labels = batch['labels']
                # Take the features from the backbone
                backout = backbone(images)
                # The RPN implementation takes as first argument the following image list
                im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
                rpnout = rpn(im_lis, backout)
                proposals=[proposal[0:keep_topK,:] for proposal in rpnout[0]]
                # A list of features produces by the backbone's FPN levels: list:len(FPN){(bz,256,H_feat,W_feat)}
                fpn_feat_list= list(backout.values())
                feature_vectors=boxHead.MultiScaleRoiAlign(fpn_feat_list,proposals)
                new_feature_vectors = torch.cat(feature_vectors, dim=0)
                labels_target, regressor_target = boxHead.create_ground_truth(proposals, labels, boxes)
                optimizer.zero_grad()
                class_logits, box_pred = boxHead(new_feature_vectors)
                loss, loss_class, loss_regr = boxHead.compute_loss(class_logits, box_pred, labels_target, regressor_target)
                loss_t_list.append(loss)
                loss_c_list.append(loss_class)
                loss_r_list.append(loss_regr)
                loss.backward()
                optimizer.step()
                print("Epoch: ", i, " Batch: ", j)
            training_loss_total.append(sum(loss_t_list) / len(loss_t_list))
            training_loss_c.append(sum(loss_c_list) / len(loss_c_list))
            training_loss_r.append(sum(loss_r_list) / len(loss_r_list))
            # Validation Time
            loss_t_list = []
            loss_c_list = []
            loss_r_list = []
            with torch.no_grad():
                for j,batch in enumerate(test_loader,0):
                    images=batch['images']
                    indexes=batch['index']
                    boxes=batch['bbox']
                    masks = batch['masks']
                    labels = batch['labels']
                    backout = backbone(images)
                    im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
                    rpnout = rpn(im_lis, backout)
                    proposals=[proposal[0:keep_topK,:] for proposal in rpnout[0]]
                    fpn_feat_list= list(backout.values())
                    feature_vectors=boxHead.MultiScaleRoiAlign(fpn_feat_list,proposals)
                    new_feature_vectors = torch.cat(feature_vectors, dim=0)
                    labels_target, regressor_target = boxHead.create_ground_truth(proposals, labels, boxes)
                    class_logits, box_pred = boxHead(new_feature_vectors)
                    loss, loss_class, loss_regr = boxHead.compute_loss(class_logits, box_pred, labels_target, regressor_target)
                    loss_t_list.append(loss)
                    loss_c_list.append(loss_class)
                    loss_r_list.append(loss_regr)
                validation_loss_total.append(sum(loss_t_list) / len(loss_t_list))
                validation_loss_c.append(sum(loss_c_list) / len(loss_c_list))
                validation_loss_r.append(sum(loss_r_list) / len(loss_r_list))
        plot_fn(training_loss_total,'Training Loss Total','Epoch','Total Loss', 'tlt')
        plot_fn(training_loss_c,'Training Loss Class','Epoch','Loss Class','tlc')
        plot_fn(training_loss_r,'Training Loss Regressor','Epoch','Loss Regr','tlr')
        plot_fn(validation_loss_total,'Validation Loss Total','Epoch','Total Loss','vlt')
        plot_fn(validation_loss_c,'Validation Loss Class','Epoch','Loss Class', 'vlc')
        plot_fn(validation_loss_r,'Validation Loss Regressor','Epoch','Loss Regr', 'vlr')
        torch.save(boxHead.state_dict(), 'model.pth')
