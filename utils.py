import numpy as np
import torch
from functools import partial
def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))

# This function computes the IOU between two set of boxes
def IOU(boxA, boxB):
    ##################################
    #TODO compute the IOU between the boxA, boxB boxes
    ##################################
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
    y_b_min = (boxBt[:,1] - boxB[:,3]/2).reshape(-1,1)
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



# This function flattens the output of the network and the corresponding anchors 
# in the sense that it concatenates  the outputs and the anchors from all the grid cells
# from all the images into 2D matrices
# Each row of the 2D matrices corresponds to a specific anchor/grid cell
# Input:
#       out_r: (bz,4,grid_size[0],grid_size[1])
#       out_c: (bz,1,grid_size[0],grid_size[1])
#       anchors: (grid_size[0],grid_size[1],4)
# Output:
#       flatten_regr: (bz*grid_size[0]*grid_size[1],4)
#       flatten_clas: (bz*grid_size[0]*grid_size[1])
#       flatten_anchors: (bz*grid_size[0]*grid_size[1],4)
def output_flattening(out_r,out_c,anchors):
    #######################################
    # TODO flatten the output tensors and anchors
    #######################################
    '''
    out_r = let's day (x,4,y,z) is it a list? or just 1 tensor coming in?
    flatten_regr = torch.zeros(2)
    now
    flatten_reger[0] = out_r[0]*out_r[2]*out_r[3]
    flatten_reger[1] = out_r[1]
    ^^^
    does that make sense?
    '''
    
    assert(out_r[0] != out_c[0],"bz is different for row and column")
    flatten_regr = out_r.view(-1,4)
    flatten_clas = out_c.view(-1,4)
    flatten_anchors = anchors.view(-1,4)

    return flatten_regr, flatten_clas, flatten_anchors




# This function decodes the output that is given in the encoded format (defined in the handout)
# into box coordinates where it returns the upper left and lower right corner of the proposed box
# Input:
#       flatten_out: (total_number_of_anchors*bz,4)
#       flatten_anchors: (total_number_of_anchors*bz,4)
# Output:
#       box: (total_number_of_anchors*bz,4)
def output_decoding(flatten_out,flatten_anchors, device='cpu'):
    #######################################
    # TODO decode the output
    #######################################
    '''
    for this it says box is same as flatten_anchors, so i am just returning:
    box = flatten_anchors
    '''
    return box
