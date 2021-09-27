CUDA_VISIBLE_DEVICES=4,5,6,7 python test.py \
--weights ./runs/train/exp57/weights/best.pt \
--data data/custom_huitou.yaml \
--batch-size  8 \
--device 4 \
--img-size 1280 \
--task test \
--verbose
