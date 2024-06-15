

python main.py --method augTrain --aug_mod geomix_1 --encoder gcn --dataset squirrel --lr 0.1 --num_layers 5 \
    --hidden_channels 128 --weight_decay 5e-4 --dropout 0.7 \
    --hops 1 --alpha 0.8 --device 1 --runs 10


python main.py --method augTrain --aug_mod geomix_2 --encoder gcn --dataset squirrel --lr 0.1 --num_layers 5 \
    --hidden_channels 128 --weight_decay 1e-3 --dropout 0.4 \
    --hops 2 --alpha 0.8 --device 1 --runs 10


python main.py --method augTrain --aug_mod geomix_3 --encoder gcn --dataset squirrel --lr 0.1 --num_layers 5 \
    --hidden_channels 128 --weight_decay 1e-4 --dropout 0.3 \
    --hops 2 --res_weight 0.8 --graph_weight 0.8 --use_weight --attn_emb_dim 16 \
    --device 1 --runs 10



python main.py --method augTrain --aug_mod geomix_1 --encoder gcn --dataset chameleon  --num_layers 2 \
    --hidden_channels 64 --lr 0.1 --weight_decay 1e-3 --dropout 0.3 \
    --hops 2 --alpha 0.8 --device 1 --runs 10


python main.py --method augTrain --aug_mod geomix_2 --encoder gcn --dataset chameleon  --num_layers 2 \
    --hidden_channels 64 --lr 0.05 --weight_decay 5e-4 --dropout 0.5 \
    --hops 2 --alpha 0.8 --device 1 --runs 10


python main.py --method augTrain --aug_mod geomix_3 --encoder gcn --dataset chameleon  --num_layers 2 \
    --hidden_channels 64 --lr 0.1 --weight_decay 5e-4 --dropout 0.4 \
    --hops 2 --res_weight 0.8 --graph_weight 0.8 --use_weight --attn_emb_dim 16 \
    --device 1 --runs 10

