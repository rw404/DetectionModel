#! /bin/sh
python3 train.py --backbone pretrained && python3 train.py --backbone pretrained --freeze True && python3 train.py --backbone pretrained --cos_scheduler False && python3 train.py --backbone pretrained --lr_backbone 0.00001 && python3 train.py --backbone pretrained --lr_backbone 0.001 && python3 train.py --backbone pretrained --lr_backbone 0.01 && cp -rf detection_exp* ../output/Pretrained/