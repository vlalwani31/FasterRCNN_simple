import numpy as np
import torch
from functools import partial
def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)

    return tuple(map(list, zip(*map_results)))

# This function compute the IOU between two set of boxes
def IOU(boxA, boxB):
    if boxA.shape == (4,):
        boxA = boxA.reshape(1,-1)
    if boxB.shape == (4,):
        boxB = boxB.reshape(1,-1)

    # compute the range of the anchor box
    x_a_min = (boxA[:,0] - boxA[:,2]/2).reshape(-1,1)
    x_a_max = (boxA[:,0] + boxA[:,2]/2).reshape(-1,1)

    y_a_min = (boxA[:,1] - boxA[:,3]/2).reshape(-1,1)
    y_a_max = (boxA[:,1] + boxA[:,3]/2).reshape(-1,1)
    # compute the anchor area
    area_A = (boxA[:,2] * boxA[:,3]).reshape(-1,1)

    # compute the range of the gt bounding box
    x_b_min = (boxB[:,0] - boxB[:,2]/2).reshape(-1,1)
    x_b_max = (boxB[:,0] + boxB[:,2]/2).reshape(-1,1)
    y_b_min = (boxB[:,1] - boxB[:,3]/2).reshape(-1,1)
    y_b_max = (boxB[:,1] + boxB[:,3]/2).reshape(-1,1)
    # compute the gt area
    area_B = (boxB[:,2] * boxB[:,3]).reshape(-1,1)

    # print(x_p_max)
    # print(x_gt_max.T)
    # print(np.minimum(x_p_max, x_gt_max.T))
    # compute the intersect w & h
    intersect_a_b_w = np.maximum(np.minimum(x_a_max, x_b_max.T) - np.maximum(x_a_min, x_b_min.T), 0)
    intersect_a_b_h = np.maximum(np.minimum(y_a_max, y_b_max.T) - np.maximum(y_a_min, y_b_min.T), 0)
    # compute intersect area
    area_intersect = intersect_a_b_w * intersect_a_b_h

    # compute union area
    area_union = (area_A + area_B.T) - area_intersect
    # print("Area of intersect: ", area_intersect)
    # print("Area of Union: ", area_union)
    iou = area_intersect / area_union
    return iou


# This function decodes the output of the box head that are given in the [t_x,t_y,t_w,t_h] format
# into box coordinates where it return the upper left and lower right corner of the bbox
# Input:
#       regressed_boxes_t: (total_proposals,4) ([t_x,t_y,t_w,t_h] format)
#       flatten_proposals: (total_proposals,4) ([x1,y1,x2,y2] format)
# Output:
#       box: (total_proposals,4) ([x1,y1,x2,y2] format)
def output_decoding(regressed_boxes_t,flatten_proposals, device='cpu'):
    box = torch.ones(regressed_boxes_t.shape)
    #print(box.shape)
    for i in range(regressed_boxes_t.shape[0]):
      regressed = regressed_boxes_t[i,:]
      proposals = flatten_proposals[i,:]
      tx = regressed[0]
      ty = regressed[1]
      tw = regressed[2]
      th = regressed[3]
      x1 = proposals[0]
      y1 = proposals[1]
      x2 = proposals[2]
      y2 = proposals[3]
      box_x = tx*x2 + x1
      box_y = ty*y2 + y1
      box_w = torch.exp(tw)*x2
      box_h = torch.exp(th)*y2
      box[i,:] = torch.tensor([box_x,box_y,box_w,box_h])

    return box
