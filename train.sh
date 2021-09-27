#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python  train.py \
#--batch-size 32 \
#--data data/custom_huitou.yaml \
#--device 1 \
#--epochs 800 \
#--workers 20 \
#--weights weights/yolov5s.pt \
#--img 640 \
#--adam



CUDA_VISIBLE_DEVICES=7  python -m torch.distributed.launch --master_port 13698 --nproc_per_node 1 train.py \
--batch-size 16 \
--data data/custom_mxx.yaml \
--device 1 \
--epochs 300 \
--workers 20 \
--weights weights/yolov5m.pt \
--img 1280 \
--adam
#--label-smoothing 0.2 \
#--augment False
