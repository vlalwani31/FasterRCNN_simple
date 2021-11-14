import torchvision
import torch
import torch.nn.functional as F
from torch import nn
from utils import *

class BoxHead(torch.nn.Module):
    def __init__(self,Classes=3,P=7):
        super(BoxHead,self).__init__()
        self.C=Classes
        self.P=P
        self.ff1 = nn.Linear(256 * P * P, 1024)
        self.ff2 = nn.Linear(1024, 1024)
        self.ff3 = nn.Linear(1024, Classes+1)
        self.ff4 = nn.Linear(1024, 4 * Classes)
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
        regressor_target=[]
        labels = []
        for bz in range(len(proposals)):
            # labels_p = torch.zeros(proposals[bz].shape[0],1)
            #number of proposals in the same image so the rows
            # for i in range(proposals[bz].shape[0]):
            #     for j in range(bbox[bz].shape[0]):
            #         box_p = proposals[bz][i,:] #all columns with 1 by 1 rows--it has x1,y1,x2,y2
            #         bbox2 = bbox[bz][j,:]#both bbox and box_p are [200,4]
            #         iou = IOU(box_p, bbox2)
            #         if iou > 0.5:
            #             labels_p[i,:] = gt_labels[bz][j]
            # labels.append(labels_p)#labels is a list rather than a tensor
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
                    # print("Proposals: ", proposals[bz].shape)
                    # print("bbox: ", bbox[bz].shape)
                    iou = IOU(props, bbox[bz])
                    r,c = iou.shape
                    max_vals, arg_max_vals = torch.max(iou, dim=1)
                    labels_p = gt_labels[bz][arg_max_vals]
                    a = max_vals < 0.5
                    labels_p[a] = 0
                    labels.append(labels_p)

                    bbox_star = torch.zeros(r,4)
                    bbox_reg = torch.zeros(r,4)
                    regressor_p = torch.zeros(r,4)
                    bbox_star[:,0] = (bbox[bz][arg_max_vals,0] + bbox[bz][arg_max_vals,2])/2 #x*
                    bbox_star[:,1] = (bbox[bz][arg_max_vals,1] + bbox[bz][arg_max_vals,3])/2 #y*
                    bbox_star[:,2] = torch.abs(bbox[bz][arg_max_vals,0] - bbox[bz][arg_max_vals,2])#w*
                    bbox_star[:,3] = torch.abs(bbox[bz][arg_max_vals,1] - bbox[bz][arg_max_vals,3])#h*

                    bbox_reg[:,0] = (props[:,0] + props[:,2])/2 #xp
                    bbox_reg[:,1] = (props[:,1] + props[:,3])/2 #yp
                    bbox_reg[:,2] = torch.abs(props[:,0] - props[:,2])#wp
                    bbox_reg[:,3] = torch.abs(props[:,1] - props[:,3])#hp

                    regressor_p[:,0] = (bbox_star[:,0]-bbox_reg[:,0])/bbox_reg[:,2] #tx
                    regressor_p[:,1] = (bbox_star[:,1]-bbox_reg[:,1])/bbox_reg[:,3] #ty
                    regressor_p[:,2] = torch.log(bbox_star[:,2]/bbox_reg[:,2])#tw
                    regressor_p[:,3] = torch.log(bbox_star[:,3]/bbox_reg[:,3])#th
                    regressor_target.append(regressor_p)
        #the regressor_target is a list rather than a tensor
        labels = torch.cat(labels, dim=0)
        regressor_target = torch.cat(regressor_target, dim=0)

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

        feature_vectors=[]
        for bz in range(len(proposals)):
        #proposal[bz].shape = 200,4
            # output = torch.zeros(proposals[bz].shape[0],256, P, P)
            box = proposals[bz] #these are x1,y1,x2,y2
            w = torch.abs(box[:,0] - box[:,2])#w* 1x200
            h = torch.abs(box[:,1] - box[:,3])#h*
            k = 4 + torch.log2(torch.sqrt(w*h)/224) #has to be 2,3,4,5 #1x200
            k = torch.clamp(k.to(torch.int), min = 2, max = 5) - 2
            # idx = [i for i in range(len(k)) if (k[i]<=5 or k[i]>=2)]
            # idx = torch.arange(len(k))
            k2 = [[],[],[],[]]
            # print("Here ", bz)
            for i in range(len(k)):
                r = 1088 / fpn_feat_list[k[i]].shape[3]
                new_x1 = proposals[bz][i,0] / r
                new_y1 = proposals[bz][i,1] / r
                new_x2 = proposals[bz][i,2] / r
                new_y2 = proposals[bz][i,3] / r
                boxe = torch.tensor([bz, new_x1, new_y1, new_x2, new_y2])
                k2[k[i]].append(boxe)
            for i in range(4):
                # print(torch.stack(k2[i], dim = 0).shape)
                if(len(k2[i]) > 0):
                    output = torchvision.ops.roi_align(fpn_feat_list[i], torch.stack(k2[i], dim = 0), 7)
                    # print(output.shape)
                    feature_vectors.append(torch.reshape(output, (-1, 256 * P * P)))
            # print("Finished ", bz)


                # output[i,:, : ,: ]= torchvision.ops.roi_align(fpn_feat_list[k[i]], boxe, (256, 7, 7))
                # output[i,:]= torchvision.ops.roi_align(fpn_feat_list[k[i]],torch.reshape(proposals[bz][i,:], (1,4)),(256*P*P))
                # if k[i] == 2:
                #     output[i,:]= torchvision.ops.roi_align(fpn_feat_list[0],torch.reshape(proposals[bz][i,:], (1,4)),(256*P*P)) #check for fpn output ask karan
                # elif k[i] == 3:
                #     output[i,:] = torchvision.ops.roi_align(fpn_feat_list[1],proposals[bz][i,:],(256*P*P)) #check for fpn output ask karan
                # elif k[i] == 4:
                #     output[i,:] = torchvision.ops.roi_align(fpn_feat_list[2],proposals[bz][i,:],(256*P*P)) #check for fpn output ask karan
                # else:
                #     output[i,:] = torchvision.ops.roi_align(fpn_feat_list[3],proposals[bz][i,:],(256*P*P)) #check for fpn output ask karan

            # feature_vectors.append(output)
            # feature_vectors.append(torch.reshape(output, (-1, 256 * P * P)))

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
        idx = torch.max(class_logits,1)[1]
        box_pred_new = torch.zeros((box_preds.shape[0],4))
        #class_logits_new = torch.zeros(class_logits.shape[0],1)

        for row in range(box_preds.shape[0]):
            if idx[row]==3:
                box_pred_new[row,:] = box_preds[row,8:12]
            elif idx[row]==2:
                box_pred_new[row,:] = box_preds[row,4:8]
            elif idx[row]==1:
                box_pred_new[row,:] = box_preds[row,0:4]
            # else:
            #     pass
        # for row in range(class_logits.shape[0]):
        #     if idx[row]==3:
        #         class_logits_new[row,:] = 3
        #     elif idx[row]==2:
        #         class_logits_new[row,:] = 2
        #     elif idx[row]==1:
        #         class_logits_new[row,:] = 1
        #     else:
        #         pass

        loss_c = self.loss_class(class_logits,labels)
        loss_r = self.loss_reg(regression_targets, box_pred_new)
        loss = loss_c + (l * loss_r)
        return loss, loss_c, loss_r

    # Compute the loss of the classifier
    def loss_class(self,class_logits,labels):
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(class_logits,labels.to(torch.long))
        return loss

    # Compute the loss of the regressor
    def loss_reg(self, regression_targets, box_pred_new):
        #torch.nn.SmoothL1Loss()
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(regression_targets, box_pred_new)
        return loss

    # Forward the pooled feature vectors through the intermediate layer and the classifier, regressor of the box head
    # Input:
    #        feature_vectors: (total_proposals, 256*P*P)
    # Outputs:
    #        class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus background, notice if you want to use
    #                                               CrossEntropyLoss you should not pass the output through softmax here)
    #        box_pred:     (total_proposals,4*C)
    def forward(self, feature_vectors):

        #TODO forward through the Intermediate layer
        X = nn.functional.relu(self.ff1(feature_vectors))
        X = nn.functional.relu(self.ff2(X))

        #TODO forward through the Classifier Head
        class_logits = self.ff3(X)

        #TODO forward through the Regressor Head
        box_pred = self.ff4(X)

        return class_logits, box_pred

#if __name__ == '__main__':
