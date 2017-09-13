#import rgbd_3det._init_paths
import sys, pdb, cv2
sys.path.append('../')
import _init_paths

import os.path as osp
import scipy.io as sio
import numpy as np
import cPickle
import matplotlib.pyplot as plt

def get_boxes_bv(props_all, im_num, con):
    # pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    #shift_path = '/data5/closerbibi/data/others/shift_size/'
    shift_path = '/home/closerbibi/workspace/tools/slicedData/shift_size/'
    sh = sio.loadmat(shift_path+'{}.mat'.format(im_num))

    boxes_bv = np.zeros((len(props_all),4))
    # 3,4,5: l(x),w(z),h(y)
    props = np.zeros((props_all.shape[0],7))
    for kk in xrange(len(props)):
        props[kk,:] = props_all[kk,con[kk]*7:con[kk]*7+7]

    boxes_bv[:,0] = props[:,0] - 0.5 * props[:,3]
    boxes_bv[:,1] = props[:,2] - 0.5 * props[:,4]
    boxes_bv[:,2] = props[:,0] + 0.5 * props[:,3]
    boxes_bv[:,3] = props[:,2] + 0.5 * props[:,4]

    boxes_bv[:,0] -= sh['xmin'][0]
    boxes_bv[:,1] -= sh['depthmin'][0]
    boxes_bv[:,2] -= sh['xmin'][0]
    boxes_bv[:,3] -= sh['depthmin'][0]

    boxes_bv = np.floor(boxes_bv*100)

    # clip box
    boxes_bv[ np.where(boxes_bv[:,0]<0) , 0] = 0 
    boxes_bv[ np.where(boxes_bv[:,1]<0) , 1] = 0 
    boxes_bv[ np.where(boxes_bv[:,2]<0) , 2] = 0 
    boxes_bv[ np.where(boxes_bv[:,3]<0) , 3] = 0 

    x_width = np.floor( (sh['xmax'][0] - sh['xmin'][0])*100)
    d_length = np.floor( (sh['depthmax'][0] - sh['depthmin'][0])*100 )
    boxes_bv[ np.where(boxes_bv[:,0]>x_width ) , 0] = x_width
    boxes_bv[ np.where(boxes_bv[:,1]>d_length) , 1] = d_length
    boxes_bv[ np.where(boxes_bv[:,2]>x_width ) , 2] = x_width
    boxes_bv[ np.where(boxes_bv[:,3]>d_length) , 3] = d_length

    # flip y
    tmp_max = d_length - boxes_bv[:, 1]
    tmp_min = d_length - boxes_bv[:, 3]
    boxes_bv[:, 1] = tmp_min
    boxes_bv[:, 3] = tmp_max

    return boxes_bv, x_width, d_length

def get_context_rois(boxes, front, x_width, d_length, ex_rt):
    # center
    cx = (boxes[:, 0] + boxes[:, 2])/2.0
    cy = (boxes[:, 1] + boxes[:, 3])/2.0
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    # new box
    xmin = cx - ex_rt*w
    xmin[np.where(xmin < 0)] = 0
    xmax = cx + ex_rt*w
    if front == True:
        xmax[np.where(xmax > 560)] = 560
    else:
        xmax[np.where(xmax > x_width)] = x_width
    ymin = cy - ex_rt*h
    ymin[np.where(ymin < 0)] = 0
    ymax = cy + ex_rt*h
    if front == True:
        ymax[np.where(ymax > 426)] = 426
    else:
        ymax[np.where(ymax > d_length)] = d_length
    boxes_new = np.vstack((xmin, ymin, xmax, ymax))
    boxes_new = boxes_new.transpose()
    return boxes_new

if __name__ == '__main__':

    # load training image list
    nyu_data_path = osp.abspath('../../dataset/NYUV2')
    with open(osp.join(nyu_data_path, 'trainval.txt')) as f:
        imlist = f.read().splitlines()

    """  data construction """
    roidb = []
    # select the first kth proposals
    num_props = 2000
    # intrinsic matrix
    from utils.help_functions import get_NYU_intrinsic_matrix
    k = get_NYU_intrinsic_matrix()

    #
    matlab_path = osp.abspath('../../matlab/NYUV2')

    for im_name in imlist:
        print(im_name)
        data = {}

        # image path
        data['image'] = osp.join(nyu_data_path, 'color', str(int(im_name)) + '.jpg')
        # depth map path (convert to [0, 255], 10m = 255)
        data['dmap'] = osp.join(matlab_path, 'dmap_f', str(int(im_name)) + '.mat')
        # bv path
        data['bvimg'] = osp.join(nyu_data_path, 'bv', 'picture_{:06d}'.format(int(im_name)) + '.jpg')

        # proposal 2d (N x 4)
        tmp = sio.loadmat(osp.join(matlab_path, 'proposal2d', str(int(im_name)) + '.mat'))
        boxes2d_prop = tmp['boxes2d_prop'].astype(np.float32)

        # rois 2d =  proposal 2d
        data['boxes'] = boxes2d_prop[0:num_props, :]
        # proposal 3d (N x 140)
        tmp = sio.loadmat(osp.join(matlab_path, 'proposal3d', str(int(im_name)) + '.mat'))
        boxes3d_prop = tmp['boxes3d_prop'].astype(np.float32)
        data['boxes_3d'] = boxes3d_prop[0:num_props, :]

        # consistency
        consistency = sio.loadmat('../../matlab/NYUV2/consistency/{:d}.mat'.format(int(im_name)))['consistency']

        # proposal bv (N x 140)
        cho_con = consistency.argmax(axis=1)
        boxes_bv, x_width, d_length = get_boxes_bv(data['boxes_3d'], im_name, cho_con[0:num_props])
        #data['boxes_bv'] = get_context_rois(boxes_bv, False, x_width, d_length, 0.75)
        data['boxes_bv'] = boxes_bv

        # scene size
        boxes = data['boxes'].copy()
        #data['rois_context'] = get_context_rois(boxes, True, x_width, d_length)

        # scene size for bv
        #data['rois_context_bv'] = get_context_rois(boxes_bv, False, x_width, d_length)


        roidb.append(data)

    print "total images: {}".format(len(roidb))

    # save training / test  data
    cache_file = 'roidb_test_19.pkl'
    with open(cache_file, 'wb') as fid:
        cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

    print "test data preparation is completed"
