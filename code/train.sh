CUDA_VISIBLE_DEVICES=7 python main.py --comment "LargeChannel_MedianDimMults_LargeBatchSize_LongEpoch" --channels 128 --dim_mults 1 2 4 --batch_size 64 --train_num_steps 40000 --n_fusion_pairs 49