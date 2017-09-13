#! /bin/bash

# save log file to 'rgbd_train.log'

python ./train_net_cmd.py --gpu 0 --solver ./models/solver-19-bn.prototxt --setType trainval \
--iters 40000 \
--weights ./models/rgbd_det_init_3d_vggNdeng-3.h5 \
2>&1 | tee ./output/rgbd_train-bvmeans.log

