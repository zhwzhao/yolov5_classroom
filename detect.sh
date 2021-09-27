CUDA_VISIBLE_DEVICES=5,6,7 python detect.py \
--weights runs/train/exp3/weights/best.pt \
--source /home/zzw/dataset/data_mxx/images_test \
--device  3 \
--img-size 1280 \
--conf-thres 0.15 \
--iou-thres 0.45
