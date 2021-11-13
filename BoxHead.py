import torch
import torch.nn.functional as F
from torch import nn
from utils import *

class BoxHead(torch.nn.Module):
    def __init__(self,Classes=3,P=7):
        self.C=Classes
        self.P=P
        # TODO initialize BoxHead



    #  This function assigns to each proposal either a ground truth box or the background class (we assume background class is 0)
    #  Input:
    #       proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #       gt_labels: list:len(bz) {(n_obj)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #  Output: (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)
    #       labels: (total_proposals,1) (the class that the proposal is assigned)
    #       regressor_target: (total_proposals,4) (target encoded in the [t_x,t_y,t_w,t_h] format)
    def create_ground_truth(self,proposals,gt_labels,bbox):
        '''
        proposals = list [bz=3, 200, 4]
        '''
        idx = []
        boxes = []
        regressor_target=[]
        labels = []
        for bz in range(len(proposals)):
            box_star = torch.zeros(bbox[bz].shape[0],4)
            box_reg = torch.zeros(proposals[bz].shape[0],4)
            regressor_p = torch.zeros(proposals[bz].shape[0],4)
            labels_p = torch.zeros(proposals[bz].shape[0],1)
            #number of proposals in the same image so the rows
            for i in range(proposals[bz].shape[0]):
                for j in range(bbox[bz].shape[0]):
                    box_p = proposals[bz][i,:] #all columns with 1 by 1 rows--it has x1,y1,x2,y2
                    bbox = bbox[bz][j,:]#both bbox and box_p are [200,4]
                    iou = utils.IOU(box_p, bbox)
                    if iou > 0.5:
                        labels_p[i,:] = gt_labels[bz][j]
            labels.append(labels_p)#labels is a list rather than a tensor

            box_star[:,0] = (bbox[:,0] + bbox[:,2])/2 #x*
            bbox_star[:,1] = (bbox[:,1] + bbox[:,3])/2 #y*
            bbox_star[:,2] = torch.abs(bbox[:,0] - bbox[:,2])#w*
            bbox_star[:,3] = torch.abs(bbox[:,1] - bbox[:,3])#h*
            
            box_reg[:,0] = (proposals[:,0] + proposals[:,2])/2 #xp
            bbox_reg[:,1] = (proposals[:,1] + proposals[:,3])/2 #yp
            bbox_reg[:,2] = torch.abs(proposals[:,0] - proposals[:,2])#wp
            bbox_reg[:,3] = torch.abs(proposals[:,1] - proposals[:,3])#hp
            
            regressor_p[:,0] = (box_star[:,0]-box_reg[:,0])/bbox_reg[:,2] #tx
            regressor_p[:,1] = (box_star[:,1]-box_reg[:,1])/bbox_reg[:,3]) #ty
            regressor_p[:,2] = torch.log(bbox_star[:,2]/bbox_reg[:,2])#tw
            regressor_p[:,3] = torch.log(bbox_star[:,3]/bbox_reg[:,3])#th
        regressor_target.append(regressor_p)
        #the regressor_target is a list rather than a tensor 
        
        return labels,regressor_target



    # This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
    # a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
    # Input:
    #      fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
    #      proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #      P: scalar
    # Output:
    #      feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
    def MultiScaleRoiAlign(self, fpn_feat_list,proposals,P=7):
        #####################################
        # Here you can use torchvision.ops.RoIAlign check the docs
        #####################################
         
        
        for bz in range(len(proposals)):
        #proposal[bz].shape = 200,4
            output = torch.zeros(proposals[bz].shape[0],256*P*P)
            for i in range(proposals[bz].shape[0]):
                box = proposals[bz][i,:] #these are x1,y1,x2,y2
                w = torch.abs(box[:,0] - box[:,2])#w*
                h = torch.abs(box[:,1] - box[:,3])#h*
                k = 4 + torch.log2(torch.sqrt(w*h)/224) 
                if k == 2:
                    roi = torchvision.ops.roi_align(fpn_feat_list[0],proposals[bz],(256*P*P)) #check for fpn output ask karan
                if k == 3:
                    roi = torchvision.ops.roi_align(fpn_feat_list[1],proposals[bz],(256*P*P)) #check for fpn output ask karan
                if k == 4:
                    roi = torchvision.ops.roi_align(fpn_feat_list[2],proposals[bz],(256*P*P)) #check for fpn output ask karan
                if k == 5:
                    roi = torchvision.ops.roi_align(fpn_feat_list[3],proposals[bz],(256*P*P)) #check for fpn output ask karan
                
            #how do i make the feature_vectors to be total_proposals,

        #loops thru the fpn list
          
                #is it by using torch.where
                #roi = roi.append(torchvision.ops.RoIAlign(fpn_feat_list[i],proposals))
        return feature_vectors



    # This function does the post processing for the results of the Box Head for a batch of images
    # Use the proposals to distinguish the outputs from each image
    # Input:
    #       class_logits: (total_proposals,(C+1))
    #       box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
    #       proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
    #       conf_thresh: scalar
    #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
    #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
    # Output:
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
    #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
    def postprocess_detections(self, class_logits, box_regression, proposals, conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=50):
        out_c = mat_clas[None, :]
        out_r = mat_coord[None, :]
        out_r_, out_c_, anchors_ = output_flattening(out_r, out_c, self.anchors)  # (bz x grid_size[0] x grid_size[1],4)
        out_bboxes = output_decoding(out_r_, anchors_)  # (bz x total anchors,4)
        img_size = (800, 1088)  # Image size
        out_bboxes[:, slice(0, 4, 2)] = np.clip(out_bboxes[:, slice(0, 4, 2)], 0, img_size[0])
        out_bboxes[:, slice(1, 4, 2)] = np.clip(out_bboxes[:, slice(1, 4, 2)], 0, img_size[1])

        sorted_indices = torch.argsort(out_c_, descending=True)
        sorted_indices = sorted_indices[:keep_num_preNMS]
        prebox = out_bboxes
        prebox = prebox[sorted_indices, :]
        clas = out_c_
        # print(clas.shape)
        # print(sorted_indices)
        clas = clas[sorted_indices]
        # apply nms
        nms_clas, nms_prebox = self.NMS(clas, prebox, IOU_thresh, keep_num_postNMS)
        return boxes, scores, labels
    
    def NMS(self,clas, prebox, thresh, keep_num_postNMS):
        ##################################
        # TODO perform NMS
        y1 = prebox[:, 0]
        x1 = prebox[:, 1]
        y2 = prebox[:, 2]
        x2 = prebox[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = torch.argsort(clas, descending=True)
        keep = []
        # while order.shape[0] > 0:
        #     i = order[0]
        #     keep.append(i)
        #     xx1 = np.maximum(x1[i], x1[order[1:]])
        #     yy1 = np.maximum(y1[i], y1[order[1:]])
        #     xx2 = np.minimum(x2[i], x2[order[1:]])
        #     yy2 = np.minimum(y2[i], y2[order[1:]])
        #     w = np.maximum(0.0, xx2 - xx1 + 1)
        #     h = np.maximum(0.0, yy2 - yy1 + 1)
        #     inter = w * h
        #     ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #     inds = np.where(ovr <= thresh)[0]
        #     order = order[(inds + 1)]
        for i in range(order.shape[0]):
            if len(keep) == 0:
                keep.append(order[i].item())
            else:
                checker = False
                for j in range(len(keep)):
                    xx1 = np.maximum(x1[order[i]], x1[keep[j]])
                    yy1 = np.maximum(y1[order[i]], y1[keep[j]])
                    xx2 = np.minimum(x2[order[i]], x2[keep[j]])
                    yy2 = np.minimum(y2[order[i]], y2[keep[j]])
                    w = np.maximum(0.0, xx2 - xx1 + 1)
                    h = np.maximum(0.0, yy2 - yy1 + 1)
                    inter = w * h
                    ovr = inter / (areas[order[i]] + areas[keep[j]] - inter)
                    if ovr > thresh:
                        checker = True
                        break
                if checker == False:
                    keep.append(order[i].item())
        keep = keep[:keep_num_postNMS]
        nms_prebox = prebox[keep,:]
        nms_clas = clas[keep]
        # print(nms_clas.shape[0])
        # print(nms_prebox)
        ##################################
        return nms_clas, nms_prebox



    # Compute the total loss of the classifier and the regressor
    # Input:
    #      class_logits: (total_proposals,(C+1)) (as outputed from forward, not passed from softmax so we can use CrossEntropyLoss)
    #      box_preds: (total_proposals,4*C)      (as outputed from forward)
    #      labels: (total_proposals,1)
    #      regression_targets: (total_proposals,4)
    #      l: scalar (weighting of the two losses)
    #      effective_batch: scalar
    # Outpus:
    #      loss: scalar
    #      loss_class: scalar
    #      loss_regr: scalar
    def compute_loss(self,class_logits, box_preds, labels, regression_targets,l=1,effective_batch=150):
        loss_class = 
        return loss, loss_class, loss_regr



    # Forward the pooled feature vectors through the intermediate layer and the classifier, regressor of the box head
    # Input:
    #        feature_vectors: (total_proposals, 256*P*P)
    # Outputs:
    #        class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus background, notice if you want to use
    #                                               CrossEntropyLoss you should not pass the output through softmax here)
    #        box_pred:     (total_proposals,4*C)
    def forward(self, feature_vectors):

        return class_logits, box_pred

if __name__ == '__main__':
