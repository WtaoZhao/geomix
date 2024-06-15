
python main.py --method augTrain --encoder geomix_1 --dataset twitch --lr 0.05 \
    --num_layers 2 --hidden_channels 32 --weight_decay 0.005 --dropout 0.6 \
    --hops 2 --alpha 0 --device 1 --runs 5


python main.py --method augTrain --encoder geomix_2 --dataset twitch --lr 0.1 \
    --num_layers 2 --hidden_channels 32 --weight_decay 0.0005 --dropout 0.5 \
    --hops 2 --alpha 0.5 --device 1 --runs 5


python main.py --method augTrain --encoder geomix_3 --dataset twitch --lr 0.1 \
    --num_layers 2 --hidden_channels 32 --weight_decay 5e-4 --dropout 0.5 \
    --hops 2 --res_weight 0.5 --graph_weight 0.5 --use_weight --attn_emb_dim 16 \
    --runs 5 --device 1

