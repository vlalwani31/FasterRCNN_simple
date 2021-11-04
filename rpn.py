import torch
from torch.nn import functional as F
from torchvision import transforms
from torch import nn, Tensor
from dataset import *
from utils import *

import torchvision


class RPNHead(torch.nn.Module):

    def __init__(self,  device='cuda', anchors_param=dict(ratio=1,scale= 320, grid_size=(50, 68), stride=16)):
        # Initialize the backbone, intermediate layer clasifier and regressor heads of the RPN
        super(RPNHead,self).__init__()

        self.device=device
        # TODO Define Backbone
        self.conv1 = nn.Conv2d(3,16,5, padding='same')
        self.batch1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,5, padding='same')
        self.batch2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,64,5, padding='same')
        self.batch3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,128,5, padding='same')
        self.batch4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128,256,5, padding='same')
        self.batch5 = nn.BatchNorm2d(256)
        self.intermediate = nn.Conv2d(256,256,3, padding='same')
        self.intermediate_batch = nn.BatchNorm2d(256)
        self.classifier = nn.Conv2d(256,1,1, padding='same')
        self.regressor = nn.Conv2d(256,4,1, padding='same')

        # TODO  Define Intermediate Layer

        # TODO  Define Proposal Classifier Head

        # TODO Define Proposal Regressor Head

        #  find anchors
        self.anchors_param=anchors_param
        self.anchors=self.create_anchors(self.anchors_param['ratio'],self.anchors_param['scale'],self.anchors_param['grid_size'],self.anchors_param['stride'])
        self.ground_dict={}





    # Forward  the input through the backbone the intermediate layer and the RPN heads
    # Input:
    #       X: (bz,3,image_size[0],image_size[1])}
    # Ouput:
    #       logits: (bz,1,grid_size[0],grid_size[1])}
    #       bbox_regs: (bz,4, grid_size[0],grid_size[1])}
    def forward(self, X):

        #TODO forward through the Backbone
        # X = nn.functional.max_pool2d(nn.functional.relu(self.batch1(self.conv1(X))), kernel_size=2, stride=2)
        # X = nn.functional.max_pool2d(nn.functional.relu(self.batch2(self.conv2(X))), kernel_size=2, stride=2)
        # X = nn.functional.max_pool2d(nn.functional.relu(self.batch3(self.conv3(X))), kernel_size=2, stride=2)
        # X = nn.functional.max_pool2d(nn.functional.relu(self.batch4(self.conv4(X))), kernel_size=2, stride=2)
        # X = nn.functional.relu(self.batch5(self.conv5(X)))
        X = self.forward_backbone(X)

        #TODO forward through the Intermediate layer
        X = nn.functional.relu(self.intermediate_batch(self.intermediate(X)))

        #TODO forward through the Classifier Head
        logits = nn.functional.sigmoid(self.classifier(X))

        #TODO forward through the Regressor Head
        bbox_regs = self.regressor(X)

        assert logits.shape[1:4]==(1,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        assert bbox_regs.shape[1:4]==(4,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return logits, bbox_regs




    # Forward input batch through the backbone
    # Input:
    #       X: (bz,3,image_size[0],image_size[1])}
    # Ouput:
    #       X: (bz,256,grid_size[0],grid_size[1])
    def forward_backbone(self,X):
        #####################################
        # TODO forward through the backbone
        X = nn.functional.max_pool2d(nn.functional.relu(self.batch1(self.conv1(X))), kernel_size=2, stride=2)
        X = nn.functional.max_pool2d(nn.functional.relu(self.batch2(self.conv2(X))), kernel_size=2, stride=2)
        X = nn.functional.max_pool2d(nn.functional.relu(self.batch3(self.conv3(X))), kernel_size=2, stride=2)
        X = nn.functional.max_pool2d(nn.functional.relu(self.batch4(self.conv4(X))), kernel_size=2, stride=2)
        X = nn.functional.relu(self.batch5(self.conv5(X)))
        #####################################
        assert X.shape[1:4]==(256,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return X



    # This function creates the anchor boxes
    # Output:
    #       anchors: (grid_size[0],grid_size[1],4)
    def create_anchors(self, aspect_ratio, scale, grid_sizes, stride):
        ######################################
        # TODO create anchors
        r,c = grid_sizes
        x_vals = (torch.arange(c) * stride) + (stride / 2)
        y_vals = (torch.arange(r) * stride) + (stride / 2)
        w_decoded = torch.sqrt(torch.tensor(aspect_ratio * scale * scale)).item()
        h_decoded = w_decoded / aspect_ratio
        anchors = torch.zeros(r, c, 4)
        for i in range(r):
            anchors[i,:,0] = x_vals
            anchors[i,:,1] = y_vals[i]
            anchors[i,:,2] = w_decoded
            anchors[i,:,3] = h_decoded

        ######################################

        assert anchors.shape == (grid_sizes[0] , grid_sizes[1],4)

        return anchors



    def get_anchors(self):
        return self.anchors



    # This function creates the ground truth for a batch of images by using
    # create_ground_truth internally
    # Input:
    #      bboxes_list: list:len(bz){(n_obj,4)}
    #      indexes:      list:len(bz)
    #      image_shape:  tuple:len(2)
    # Output:
    #      ground_clas: (bz,1,grid_size[0],grid_size[1])
    #      ground_coord: (bz,4,grid_size[0],grid_size[1])
    def create_batch_truth(self,bboxes_list,indexes,image_shape):
        #####################################
        # TODO create ground truth for a batch of images
        g_class_list = []
        g_coord_list = []
        for i in range(len(indexes)):
            cl_out, co_out = self.create_ground_truth(bboxes_list[i], indexes[i], self.anchors_param['grid_size'], self.anchors, image_shape)
            g_class_list.append(cl_out)
            g_coord_list.append(co_out)
        ground_clas = torch.stack(g_class_list, dim = 0)
        ground_coord = torch.stack(g_coord_list, dim = 0)
        #####################################
        assert ground_clas.shape[1:4]==(1,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        assert ground_coord.shape[1:4]==(4,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return ground_clas, ground_coord


    # This function creates the ground truth for one image
    # It also caches the ground truth for the image using its index
    # Input:
    #       bboxes:      (n_boxes,4)
    #       index:       scalar (the index of the image in the total dataset used for caching)
    #       grid_size:   tuple:len(2)
    #       anchors:     (grid_size[0],grid_size[1],4)
    # Output:
    #       ground_clas:  (1,grid_size[0],grid_size[1])
    #       ground_coord: (4,grid_size[0],grid_size[1])
    def create_ground_truth(self, bboxes, index, grid_size, anchors, image_size):
        key = str(index)
        if key in self.ground_dict:
            groundt, ground_coord = self.ground_dict[key]
            return groundt, ground_coord

        #####################################################
        # TODO create ground truth for a single image
        print(bboxes)
        ground_clas = torch.zeros(1, grid_size[0], grid_size[1]) + 0.5
        ground_coord = torch.zeros(4,grid_size[0],grid_size[1])
        anch = anchors.view(-1,4)
        t_bboxes = torch.zeros_like(bboxes)
        y_b_1 = bboxes[:,0]
        y_b_2 = bboxes[:,2]
        x_b_1 = bboxes[:,1]
        x_b_2 = bboxes[:,3]
        t_bboxes[:,0] = (x_b_2 + x_b_1) / 2
        t_bboxes[:,1] = (y_b_2 + y_b_1) / 2
        t_bboxes[:,2] = x_b_2 - x_b_1
        t_bboxes[:,3] = y_b_2 - y_b_1
        iou_results = IOU(anch, t_bboxes)
        iou_results = iou_results.view(grid_size[0],grid_size[1], len(bboxes))
        max_vals, max_ind = torch.max(iou_results, 2)
        ground_clas[0,max_vals < 0.3] = 0
        a = max_vals > 0.7
        buf1 = int(anchors[0,0,2] / (2*self.anchors_param['stride']))
        buf2 = int(anchors[0,0,3] / (2*self.anchors_param['stride']))
        for i in range(len(bboxes)):
            b = torch.max(iou_results[buf1:(grid_size[0]-buf1),buf2:(grid_size[1]-buf2),i])
            c = iou_results[:,:,i] == b
            # a[c] = True
            ground_clas[0, c] = 1
            ground_coord[0,c] = (t_bboxes[i,0] - anchors[c,:][:,0]) / anchors[c,:][:,2]
            ground_coord[1,c] = (t_bboxes[i,1] - anchors[c,:][:,1]) / anchors[c,:][:,3]
            ground_coord[2,c] = torch.log(t_bboxes[i,2]) - torch.log(anchors[c,:][:,2])
            ground_coord[3,c] = torch.log(t_bboxes[i,3]) - torch.log(anchors[c,:][:,3])
        # anch_x1_check = anchors[:,:,0] - (anchors[:,:,2] / 2)
        # anch_x2_check = anchors[:,:,0] + (anchors[:,:,2] / 2)
        # anch_y1_check = anchors[:,:,1] - (anchors[:,:,3] / 2)
        # anch_y2_check = anchors[:,:,1] + (anchors[:,:,3] / 2)
        # # Remove anchors with cross boundary values
        # print(anch_x1_check.shape)
        # d = (anch_x1_check < 0).nonzero()
        # print(d.shape) #or (anch_x1_check > image_size[0]) or (anch_x2_check < 0) or (anch_x2_check > image_size[0]) or (anch_y1_check < 0) or (anch_y1_check > image_size[1]) or (anch_y2_check < 0) or (anch_y2_check > image_size[1])).nonzero()
        # a[d] = False
        # d = (anch_x1_check > image_size[0]).nonzero()
        # print(d.shape)
        # a[d] = False
        if (ground_clas[0,a].shape[0] != 0):
            ground_clas[0,a] = 1
            ground_coord[0,a] = (t_bboxes[max_ind[a],0] - anchors[a,:][:,0]) / anchors[a,:][:,2]
            ground_coord[1,a] = (t_bboxes[max_ind[a],1] - anchors[a,:][:,1]) / anchors[a,:][:,3]
            ground_coord[2,a] = torch.log(t_bboxes[max_ind[a],2]) - torch.log(anchors[a,:][:,2])
            ground_coord[3,a] = torch.log(t_bboxes[max_ind[a],3]) - torch.log(anchors[a,:][:,3])

        for i in range(grid_size[0]):
            ground_clas[0, i, :buf2] = 0.5
            ground_clas[0, i, (grid_size[1]-buf2+1):] = 0.5
        for i in range(grid_size[1]):
            ground_clas[0, :buf1, i] = 0.5
            ground_clas[0, (grid_size[0]-buf1+1):, i] = 0.5

        #####################################################

        self.ground_dict[key] = (ground_clas, ground_coord)

        assert ground_clas.shape==(1,grid_size[0],grid_size[1])
        assert ground_coord.shape==(4,grid_size[0],grid_size[1])

        return ground_clas, ground_coord





    # Compute the loss of the classifier
    # Input:
    #      p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
    #      n_out:     (negatives_on_mini_batch) (output of the classifier for sampled anchors with negative gt labels
    def loss_class(self,p_out,n_out):

        #torch.nn.BCELoss()
        # TODO compute classifier's loss
        criterion = torch.nn.BCELoss()
        p_targ = torch.ones_like(p_out)
        n_targ = torch.zeros_like(n_out)
        loss1 = criterion(p_out, p_targ)
        loss2 = criterion(n_out, n_targ)
        loss = (loss1 + loss2) / 2
        return loss



    # Compute the loss of the regressor
    # Input:
    #       pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
    #       pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
    def loss_reg(self, pos_target_coord, pos_out_r):
        #torch.nn.SmoothL1Loss()
        # TODO compute regressor's loss
        criterion = torch.nn.BCELoss()
        loss = criterion(pos_out_r, pos_target_coord)
        return loss



    # Compute the total loss
    # Input:
    #       clas_out: (bz,1,grid_size[0],grid_size[1])
    #       regr_out: (bz,4,grid_size[0],grid_size[1])
    #       targ_clas:(bz,1,grid_size[0],grid_size[1])
    #       targ_regr:(bz,4,grid_size[0],grid_size[1])
    #       l: lambda constant to weight between the two losses
    #       effective_batch: the number of anchors in the effective batch (M in the handout)
    def compute_loss(self, clas_out, regr_out, targ_clas, targ_regr, l=1, effective_batch=50):
        #############################
        # TODO compute the total loss
        # Flatten the inputs
        regr_out_flat, clas_out_flat, anch_flat = output_flattening(regr_out, clas_out, self.anchors)
        targ_regr_flat, targ_clas_flat, _ = output_flattening(targ_regr, targ_clas, self.anchors)
        # Check where target has positve labels
        b = (targ_clas_flat == 1)
        indexes_p = b.nonzero()
        # Check where target has negative labels
        c = (targ_clas_flat == 0)
        indexes_n = c.nonzero()
        p_out = 0
        n_out = 0
        pos_out_r = 0
        pos_target_coord = 0
        len_indexes_p = indexes_p.size(dim=0)
        # Check if you have enough postive labels
        if (len_indexes_p < (effective_batch / 2)):
            # If not enough positive labels, then get all postive labels
            # And sample remaining postions as negative labels
            p_out = clas_out_flat[indexes_p]
            pos_out_r = regr_out_flat[indexes_p, :]
            pos_target_coord = targ_regr_flat[indexes_p, :]
            n_indexes = torch.randperm(indexes_n.size(dim=0))[:(effective_batch - len_indexes_p)]
            n_out = clas_out_flat[indexes_n[n_indexes]]
        else:
            # If yes, then sample M/2 postive labels and negative labels
            p_indexes = torch.randperm(indexes_p.size(dim=0))[:(effective_batch / 2)]
            p_out = clas_out_flat[indexes_p[p_indexes]]
            pos_out_r = regr_out_flat[indexes_p[p_indexes], :]
            pos_target_coord = targ_regr_flat[indexes_p[p_indexes], :]
            n_indexes = torch.randperm(indexes_n.size(dim=0))[:(effective_batch / 2)]
            n_out = clas_out_flat[indexes_n[n_indexes]]
        loss_c = self.loss_class(p_out, n_out)
        loss_r = self.loss_reg(pos_target_coord, pos_out_r)
        loss = loss_c + (l * loss_r)
        #############################
        return loss, loss_c, loss_r



    # Post process for the outputs for a batch of images
    # Input:
    #       out_c:  (bz,1,grid_size[0],grid_size[1])}
    #       out_r:  (bz,4,grid_size[0],grid_size[1])}
    #       IOU_thresh: scalar that is the IOU threshold for the NMS
    #       keep_num_preNMS: number of masks we will keep from each image before the NMS
    #       keep_num_postNMS: number of masks we will keep from each image after the NMS
    # Output:
    #       nms_clas_list: list:len(bz){(Post_NMS_boxes)} (the score of the boxes that the NMS kept)
    #       nms_prebox_list: list:len(bz){(Post_NMS_boxes,4)} (the coordinates of the boxes that the NMS kept)
    def postprocess(self,out_c,out_r, IOU_thresh=0.5, keep_num_preNMS=50, keep_num_postNMS=10):
       ####################################
       # TODO postprocess a batch of images
       batch_size = out_c.shape[0]
       # K, N = keep_num_preNMS, keep_num_postNMS
       out_r_,out_c_,anchors_ = utils.output_flattening(out_r,out_c,self.anchor) #(bz x grid_size[0] x grid_size[1],4)
       out_bboxes = utils.output_decoding(out_r_,anchors_) #(bz x total anchors,4)
       img_size = (800, 1088)  # Image size
       out_bboxes[:, slice(0, 4, 2)] = np.clip(out_bboxes[:, slice(0, 4, 2)], 0, img_size[0])
       out_bboxes[:, slice(1, 4, 2)] = np.clip(out_bboxes[:, slice(1, 4, 2)], 0, img_size[1])
       steps = out_c_.shape[0]/batch_size
       batches = list(range(0, (out_c_.shape[0] + steps), steps))
       nms_clas_list = []
       nms_prebox_list = []
       for i in range(len(batches)):
           if i<(len(batch_size)-1):
               sorted_indices = torch.argsort(out_c_[batches[i]:batches[i+1],:],descending=True)
               sorted_indices = sorted_indices[:keep_num_preNMS]
               prebox = out_bboxes[batches[i]:batches[i+1],:]
               prebox = prebox[sorted_indices,:]
               clas = out_c_[batches[i]:batches[i+1],:]
               clas = clas[sorted_indices,:]
               #apply nms
               nms_clas,nms_prebox = NMS(clas, prebox, IOU_thresh, keep_num_postNMS)
               nms_prebox_list.append(nms_prebox)
               nms_clas_list.append(nms_clas)
           else:
               pass

       #####################################
       return nms_clas_list, nms_prebox_list



    # Post process the output for one image
    # Input:
    #      mat_clas: (1,grid_size[0],grid_size[1])}  (scores of the output boxes)
    #      mat_coord: (4,grid_size[0],grid_size[1])} (encoded coordinates of the output boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4) (decoded coordinates of the boxes that the NMS kept)
    def postprocessImg(self,mat_clas,mat_coord, IOU_thresh,keep_num_preNMS, keep_num_postNMS):
            ######################################
            # TODO postprocess a single image
            out_c = mat_clas[None, :]
            out_r = mat_coord[None, :]
            out_r_, out_c_, anchors_ = utils.output_flattening(out_r, out_c, self.anchor)  # (bz x grid_size[0] x grid_size[1],4)
            out_bboxes = utils.output_decoding(out_r_, anchors_)  # (bz x total anchors,4)
            img_size = (800, 1088)  # Image size
            out_bboxes[:, slice(0, 4, 2)] = np.clip(out_bboxes[:, slice(0, 4, 2)], 0, img_size[0])
            out_bboxes[:, slice(1, 4, 2)] = np.clip(out_bboxes[:, slice(1, 4, 2)], 0, img_size[1])

            sorted_indices = torch.argsort(out_c_, descending=True)
            sorted_indices = sorted_indices[:keep_num_preNMS]
            prebox = out_bboxes
            prebox = prebox[sorted_indices, :]
            clas = out_c_
            clas = clas[sorted_indices, :]
            # apply nms
            nms_clas, nms_prebox = NMS(clas, prebox, IOU_thresh, keep_num_postNMS)
            #####################################

            return nms_clas, nms_prebox



    # Input:
    #       clas: (top_k_boxes) (scores of the top k boxes)
    #       prebox: (top_k_boxes,4) (coordinate of the top k boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
    def NMS(self,clas, prebox, thresh, keep_num_postNMS):
        ##################################
        # TODO perform NMS
        y1 = prebox[:, 0]
        x1 = prebox[:, 1]
        y2 = prebox[:, 2]
        x2 = prebox[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = clas.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        keep = keep[:keep_num_postNMS]
        nms_prebox = prebox[keep,:]
        nms_clas = clas[keep,:]
        ##################################
        return nms_clas, nms_prebox

# if __name__=="__main__":
