#import rgbd_3det._init_paths
import sys, pdb
sys.path.append('..')
import _init_paths
import caffe

# set up caffe gpu
caffe.set_mode_gpu()
caffe.set_device(1)

''' init network from pretrained vgg model'''

net_vgg16 = caffe.Net('./models/VGG16/test.prototxt',
                      './models/VGG16/vgg16_fast_rcnn_iter_40000.caffemodel', caffe.TEST)

net_deng = caffe.Net('/data5/closerbibi/repo/caffe-repo/Amodal3Det/rgbd_3det/models/test-19-bn.prototxt',
                      '/data5/closerbibi/repo/caffe-repo/Amodal3Det/rgbd_3det/output/concat/rgbd_3det_iter_40000.h5', caffe.TEST)

solver_prototxt = './models/solver-19-bn.prototxt'
solver = caffe.SGDSolver(solver_prototxt)
net = solver.net


params = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
          'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
params_d = ['conv1_1d', 'conv1_2d', 'conv2_1d', 'conv2_2d', 'conv3_1d', 'conv3_2d', 'conv3_3d',
          'conv4_1d', 'conv4_2d', 'conv4_3d', 'conv5_1d', 'conv5_2d', 'conv5_3d']

# copy weights from vgg16
for name in params:
    #if name != 'conv1_1':
    net.params[name +'bv'][0].data[...] = net_vgg16.params[name][0].data
    net.params[name +'bv'][1].data[...] = net_vgg16.params[name][1].data

# copy weights from deng
for name in params:
    net.params[name][0].data[...] = net_deng.params[name][0].data
    net.params[name][1].data[...] = net_deng.params[name][1].data

for name in params_d:
    net.params[name][0].data[...] = net_deng.params[name][0].data
    net.params[name][1].data[...] = net_deng.params[name][1].data

pdb.set_trace()
print(net.params['bbox_pred_3d'][0].data)
print(net.params['bbox_pred_3d'][1].data)

print(net.params['cls_score'][0].data)
print(net.params['cls_score'][1].data)

print(net.params['fc6'][0].data)
print(net.params['fc6'][1].data)

print(net.params['fc7'][0].data)
print(net.params['fc7'][1].data)

#net.save('./models/rgbd_det_init_3d_allvgg-3.caffemodel')
# for tensor > 2 GB
net.save_hdf5('./models/rgbd_det_init_3d_vggNdeng-3.h5')
