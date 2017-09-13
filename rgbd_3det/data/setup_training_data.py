
"""
  set up training set for 19 classes object detection
"""
#import rgbd_3det._init_paths
import sys, pdb, cv2
sys.path.append('../')
import _init_paths

import os.path as osp
import scipy.io as sio
from cnn.config import cfg
import PIL
import cPickle
import numpy as np
import math
import matplotlib.pyplot as plt

def debug_rois(bv_props, props_2d, impath_2d, bvimpath):
    im = cv2.imread(impath_2d)
    im = im[:, :, (2, 1, 0)]
    bvimg = cv2.imread(bvimpath)
    bvimg = bvimg[:, :, (2, 1, 0)]

    for i in xrange(3):
        plt.figure(1)
        x1, y1, x2, y2 = props_2d[i,0], props_2d[i,1], props_2d[i,2], props_2d[i,3]
        plt.gca().add_patch(
            plt.Rectangle((x1, y1),
                           x2 - x1, 
                           y2 - y1, fill=False,
                           edgecolor='g', linewidth=2.0)
        )
        plt.imshow(im);
        plt.figure(2)
        x1, y1, x2, y2 = bv_props[i,0], bv_props[i,1], bv_props[i,2], bv_props[i,3]
        plt.gca().add_patch(
            plt.Rectangle((x1, y1),
                           x2 - x1,
                           y2 - y1, fill=False,
                           edgecolor='g', linewidth=2.0)
        )
        plt.imshow(bvimg); plt.show()



def get_boxes_bv(props_all, im_num, gt_boxes_3d, con):
    """
    Args:
        props_all: N x 7
        im_num: image index

    Returns: boxes_bv  N x 4

    """
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

    boxes_bv = np.vstack(( np.zeros((gt_boxes_3d.shape[0],4)) , boxes_bv))

    return boxes_bv, x_width, d_length

def get_boxes_3d(gt_boxes_3d, prop_boxes, max_classes, consistency):
    """
    Args:
        gt_boxes_3d: N x 7
        prop_boxes: N x 140
        max_classes: class label start from 0

    Returns: boxes_3d  N x 7

    """
    n_boxes = len(max_classes)
    n_gt_boxes = gt_boxes_3d.shape[0]
    n_prop_boxes = prop_boxes.shape[0]
    assert(n_boxes == (n_gt_boxes + n_prop_boxes))

    cls = max_classes[n_gt_boxes:]
    props = np.zeros((0, 7), dtype=prop_boxes.dtype)
    for i in xrange(len(cls)):
        label = cls[i]
        sid = label*7
        eid = sid + 7
        tmp = prop_boxes[i, sid:eid]
        props = np.vstack((props, tmp))


    # stacking with gt box
    boxes_3d = np.vstack((gt_boxes_3d, props))

    # making proposals of bird's view
    cho_con = consistency.argmax(axis=1)
    boxes_bv, x_width, d_length = get_boxes_bv(prop_boxes, im_name, gt_boxes_3d, cho_con)

    return boxes_3d, boxes_bv, x_width, d_length


def flip_boxes_3d(boxes_3d, K, width):

    # theta
    boxes_3d[:, -1] = -boxes_3d[:, -1]

    # cx
    cx = boxes_3d[:, 0]
    cz = boxes_3d[:, 2]
    ox = K[0, 2]
    fx = K[0, 0]
    x = cx*fx/(cz + cfg.EPS) + ox

    # flip x (x is start from 1)
    x1 = width - x + 1
    boxes_3d[:, 0] = (x1-ox)*cz/fx

    return

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

    # pre-defined 19 classes plus background
    classes = tuple(cfg.classes)

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

    matlab_path = osp.abspath('../../matlab/NYUV2')
    for im_name in imlist:
        print(im_name)
        data = {}

        ''' --------------------ground truth----------------------------------- '''
        # gt boxes 3d: [x,y,z,l,w,h,theta]
        tmp = sio.loadmat(osp.join(matlab_path, 'gt_3D_19', str(int(im_name)) + '.mat'))
        gt_boxes_3d = tmp['gt_boxes_3d'].astype(np.float32)
        if gt_boxes_3d.shape[0] == 0:
            print 'no gt for target objects and skip.'
            continue

        # gt2Dsel [xmin, ymin, xmax, ymax]
        tmp = sio.loadmat(osp.join(matlab_path, 'gt_2Dsel_19', str(int(im_name)) + '.mat'))
        gt_boxes_sel = tmp['gt_boxes_sel'].astype(np.float32)
        num_gt_boxes = gt_boxes_sel.shape[0]

        # gt class ids
        tmp = sio.loadmat(osp.join(matlab_path, 'gt_label_19', str(int(im_name)) + '.mat'))
        gt_class_labels = tmp['gt_class_ids'].astype(np.float32)

        '''---------------------------proposals--------------------------------- '''
        # proposal 2d (N x 4)
        tmp = sio.loadmat(osp.join(matlab_path, 'proposal2d', str(int(im_name)) + '.mat'))
        try:
            boxes2d_prop = tmp['boxes2d_prop'].astype(np.float32)
        except:
            pdb.set_trace()

        # proposal 3d (N x 140)
        tmp = sio.loadmat(osp.join(matlab_path, 'proposal3d', str(int(im_name)) + '.mat'))
        boxes3d_prop = tmp['boxes3d_prop'].astype(np.float32)

        '''---------------------------inputs----------------------------------- '''
        # image path
        data['image'] = osp.join(nyu_data_path, 'color', str(int(im_name)) + '.jpg')
        # depth map path (convert to [0, 255], 10m = 255)
        data['dmap'] = osp.join(matlab_path, 'dmap_f', str(int(im_name)) + '.mat')
        # bv image path
        data['bvimg'] = osp.join(nyu_data_path, 'bv', 'picture_{:06d}'.format(int(im_name)) + '.jpg')
        # HHA
        # data['dmap'] = osp.join(nyu_data_path, 'HHA', str(int(im_name)) + '.png')

        # rois 2d = gt2Dsel + proposal 2d
        data['boxes'] = np.vstack((gt_boxes_sel, boxes2d_prop[0:num_props-num_gt_boxes, :]))

        # overlap: compare rois proposals with gt_rois_selection
        tmp = sio.loadmat(osp.join(matlab_path, 'gt_overlaps_19', str(int(im_name)) + '.mat'))
        data['gt_overlaps'] = tmp['gt_overlaps'][0:num_props, :]
        gt_overlaps = data['gt_overlaps']

        # max_classes and max_overlaps
        max_overlaps = gt_overlaps.max(axis=1)
        max_classes = gt_overlaps.argmax(axis=1)
        data['max_classes'] = max_classes
        data['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)

        # consistency
        b3d_proj = sio.loadmat('../../matlab/NYUV2/consistency/{:d}.mat'.format(int(im_name)))['b3d_proj']
        consistency = sio.loadmat('../../matlab/NYUV2/consistency/{:d}.mat'.format(int(im_name)))['consistency']

        pdb.set_trace()
        # rois 3d (N x 7) = gt boxes 3d + proposal 3d
        # gabriel: add boxes of bird's eye view
        data['boxes_3d'], tmp_bv, x_width, d_length = \
            get_boxes_3d(gt_boxes_3d, boxes3d_prop[0:num_props-num_gt_boxes, :], max_classes, consistency[0:num_props-num_gt_boxes, :])

        #data['boxes_bv'] = get_context_rois(tmp_bv, False, x_width, d_length, 0.75)
        data['boxes_bv'] = tmp_bv

        # flipped
        data['flipped'] = False

        # context, gabriel
        boxes = data['boxes'].copy()
        boxes_bv = data['boxes_bv'].copy()

        roidb.append(data)

        #debug_rois(boxes_bv[len(gt_boxes_3d):], boxes[len(gt_boxes_3d):], data['image'], data['bvimg'])

    """ Data augmentation """
    num_images = len(roidb)

    if cfg.TRAIN.USE_FLIPPED:
        widths = [PIL.Image.open(roidb[i]['image']).size[0]
                  for i in xrange(num_images)]
        # gabriel
        widths_bv = [PIL.Image.open(roidb[i]['bvimg']).size[0]
                  for i in xrange(num_images)]
        print('flipping...')
        for i in xrange(num_images):
            print '{}image'.format(i)
            # flip 2d boxes
            boxes = roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()

            # flip 3d boxes
            boxes_3d = roidb[i]['boxes_3d'].copy()
            flip_boxes_3d(boxes_3d, k, widths[i])

            # flip bv boxes, gabriel
            boxes_bv = roidb[i]['boxes_bv'].copy()
            oldx1_bv = boxes_bv[:, 0].copy()
            oldx2_bv = boxes_bv[:, 2].copy()
            boxes_bv[:, 0] = widths_bv[i] - oldx2_bv - 1
            boxes_bv[:, 2] = widths_bv[i] - oldx1_bv - 1
            try:
                assert (boxes_bv[:, 2] >= boxes_bv[:, 0]).all()
            except:
                pdb.set_trace()

            entry = {'image': roidb[i]['image'],
                     'bvimg': roidb[i]['bvimg'], # gabriel
                     'boxes' : boxes,
                     'gt_overlaps' : roidb[i]['gt_overlaps'],
                     'flipped' : True,
                     'max_classes': roidb[i]['max_classes'],
                     'max_overlaps': roidb[i]['max_overlaps'],
                     'dmap': roidb[i]['dmap'],
                     'boxes_3d': boxes_3d,
                     'boxes_bv': boxes_bv # gabriel
                     }
                     #'rois_context': boxes_context,
                     #'rois_context_bv': boxes_context_bv # gabriel
                     #}

            roidb.append(entry)

    print "total images: {}".format(len(roidb))

    print "all keys: {}".format(roidb[0].keys())
    # save training / test  data
    cache_file = 'roidb_trainval_19.pkl'
    with open(cache_file, 'wb') as fid:
        cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

    print "training data preparation is completed"
