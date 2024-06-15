# 10 to 30


python transfer.py --method augTrain --encoder geomix_1 --src_pu 10 --tgt_pu 30 --src_event 10 --tgt_event 20 \
    --dataset pileup --labeled_ratio 0.2 --batch_size 5 --lr 0.005 --num_layers 2 --hidden_channels 64 \
    --weight_decay 1e-3 --dropout 0.3 --hops 2 --alpha 0 --aug_lamb 10 \
    --runs 5 --epochs 200 --device 1 


python transfer.py --method augTrain --encoder geomix_2 --src_pu 10 --tgt_pu 30 --src_event 10 --tgt_event 20 \
    --dataset pileup --labeled_ratio 0.2 --batch_size 5 --lr 0.005 --num_layers 2 --hidden_channels 64 \
    --weight_decay 5e-3 --dropout 0.5 --device 3 --hops 3 --alpha 0.1 \
    --runs 5 --epochs 200


python transfer.py --method augTrain --encoder geomix_3 --src_pu 10 --tgt_pu 30 --src_event 10 --tgt_event 20 \
    --dataset pileup --labeled_ratio 0.2 --batch_size 5 --lr 0.005 --num_layers 2 --hidden_channels 64 \
    --weight_decay 5e-3 --dropout 0.5 --device 3 --hops 2 --res_weight 0.1 --graph_weight 0.5 --use_weight --attn_emb_dim 16 \
    --runs 5 --epochs 200

# 30 to 10 


python transfer.py --method augTrain --encoder geomix_1 --src_pu 30 --tgt_pu 10 --src_event 20 --tgt_event 10 \
    --dataset pileup --labeled_ratio 0.2 --batch_size 5 --lr 0.001 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.005 --dropout 0.5 --device 1 --alpha 0.8 --hops 1 \
    --runs 5 --epochs 200


python transfer.py --method augTrain --encoder geomix_2 --src_pu 30 --tgt_pu 10 --src_event 20 --tgt_event 10 \
    --dataset pileup --labeled_ratio 0.2 --batch_size 5 --lr 0.001 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.005 --dropout 0.5 --device 3 --alpha 0.8 --hops 1 \
    --runs 5 --epochs 200


python transfer.py --method augTrain --encoder geomix_3 --src_pu 30 --tgt_pu 10 --src_event 20 --tgt_event 10 \
    --dataset pileup --labeled_ratio 0.2 --batch_size 5 --lr 0.001 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.005 --dropout 0.5 --device 3 --hops 1 --res_weight 0.8 --graph_weight 0.5 --use_weight --attn_emb_dim 16 \
    --runs 5 --epochs 200


# 30 to 50


python transfer.py --method augTrain --encoder geomix_1 --src_pu 30 --tgt_pu 50 --src_event 10 --tgt_event 20 \
    --dataset pileup --labeled_ratio 0.2 --batch_size 5 --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 1e-2 --dropout 0.5 --device 2 --alpha 0.1 --hops 3 --label_grad \
    --runs 5 --epochs 200



python transfer.py --method augTrain --encoder geomix_2 --src_pu 30 --tgt_pu 50 --src_event 10 --tgt_event 20 \
    --dataset pileup --labeled_ratio 0.2 --batch_size 5 --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 1e-2 --dropout 0.5 --device 2 --alpha 0.1 --hops 3 --label_grad \
    --runs 5 --epochs 200


python transfer.py --method augTrain --encoder geomix_3 --src_pu 30 --tgt_pu 50 --src_event 10 --tgt_event 20 \
    --dataset pileup --labeled_ratio 0.2 --batch_size 5 --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 1e-2 --dropout 0.3 --device 2 --res_weight 0.1 --hops 3 --use_weight --label_grad \
    --runs 5 --epochs 200


# 50 to 30


python transfer.py --method augTrain --encoder geomix_1 --src_pu 50 --tgt_pu 30 --src_event 10 --tgt_event 20 \
    --dataset pileup --labeled_ratio 0.2 --batch_size 5 --lr 0.005 --num_layers 2 --hidden_channels 64 \
    --weight_decay 1e-4 --dropout 0.6 --device 3 --alpha 0.1 --hops 2 --label_grad \
    --runs 5 --epochs 200


python transfer.py --method augTrain --encoder geomix_2 --src_pu 50 --tgt_pu 30 --src_event 10 --tgt_event 20 \
    --dataset pileup --labeled_ratio 0.2 --batch_size 5 --lr 0.005 --num_layers 2 --hidden_channels 64 \
    --weight_decay 1e-4 --dropout 0.6 --device 3 --alpha 0.1 --hops 2 --label_grad \
    --runs 5 --epochs 200


python transfer.py --method augTrain --encoder geomix_3 --src_pu 50 --tgt_pu 30 --src_event 10 --tgt_event 20 \
    --dataset pileup --labeled_ratio 0.2 --batch_size 5 --lr 0.005 --num_layers 2 --hidden_channels 64 \
    --weight_decay 1e-4 --dropout 0.6 --device 3 --res_weight 0.1 --hops 2 --use_weight --label_grad \
    --runs 5 --epochs 200


# gg to qq


python transfer.py --method augTrain --encoder geomix_1 --src_pu 30 --tgt_pu 30 --src_event 10 --tgt_event 20 --src_sig gg \
    --tgt_sig qq --dataset pileup --labeled_ratio 0.2 --batch_size 5 --lr 0.05 --num_layers 2 --hidden_channels 64 \
    --weight_decay 1e-3 --dropout 0.5 --device 3 --hops 2 --alpha 0 --label_grad \
    --runs 5 --epochs 200


python transfer.py --method augTrain --encoder geomix_2 --src_pu 30 --tgt_pu 30 --src_event 10 --tgt_event 20 --src_sig gg \
    --tgt_sig qq --dataset pileup --labeled_ratio 0.2 --batch_size 5 --lr 0.05 --num_layers 2 --hidden_channels 64 \
    --weight_decay 1e-3 --dropout 0.5 --device 2 --hops 2 --alpha 0.1 --label_grad \
    --runs 5 --epochs 200


python transfer.py --method augTrain --encoder geomix_3 --src_pu 30 --tgt_pu 30 --src_event 10 --tgt_event 20 --src_sig gg \
    --tgt_sig qq --dataset pileup --labeled_ratio 0.2 --batch_size 5 --lr 0.05 --num_layers 2 --hidden_channels 64 \
    --weight_decay 5e-3 --dropout 0.5 --device 3 --hops 2 --res_weight 0.1 --use_weight --label_grad \
    --runs 5 --epochs 200

# qq to gg


python transfer.py --method augTrain --encoder geomix_1 --src_pu 30 --tgt_pu 30 --src_event 10 --tgt_event 20 --src_sig qq \
    --tgt_sig gg --dataset pileup --labeled_ratio 0.2 --batch_size 5 --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 5e-4 --dropout 0.5 --device 2 --hops 2 --alpha 0 --label_grad \
    --runs 5 --epochs 200


python transfer.py --method augTrain --encoder geomix_2 --src_pu 30 --tgt_pu 30 --src_event 10 --tgt_event 20 --src_sig qq \
    --tgt_sig gg --dataset pileup --labeled_ratio 0.2 --batch_size 5 --lr 0.005 --num_layers 2 --hidden_channels 64 \
    --weight_decay 1e-3 --dropout 0.5 --device 3 --hops 2 --alpha 0.1 --label_grad \
    --runs 5 --epochs 200


python transfer.py --method augTrain --encoder geomix_3 --src_pu 30 --tgt_pu 30 --src_event 10 --tgt_event 20 --src_sig qq \
    --tgt_sig gg --dataset pileup --labeled_ratio 0.2 --batch_size 5 --lr 0.005 --num_layers 2 --hidden_channels 64 \
    --weight_decay 1e-3 --dropout 0.5 --device 3 --hops 2 --res_weight 0.1 --use_weight --graph_weight 0.8 --label_grad \
    --runs 5 --epochs 200
