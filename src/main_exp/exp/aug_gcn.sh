
python main.py --method augTrain --encoder geomix_1 --dataset cora --lr 0.1 \
    --num_layers 2 --hidden_channels 16 --weight_decay 5e-4 --dropout 0.5 \
    --device 1 --alpha 0.1 --hops 3 --label_grad --runs 5 


python main.py --method augTrain --encoder geomix_2 --dataset cora --lr 0.1 \
    --num_layers 2 --hidden_channels 16 --weight_decay 5e-4 --dropout 0.5 \
    --device 1 --hops 2 --alpha 0.1 --label_grad --runs 5


python main.py --method augTrain --encoder geomix_3 --dataset cora --lr 0.1 \
    --num_layers 2 --hidden_channels 16 --weight_decay 5e-4 --dropout 0.5 \
    --hops 2 --res_weight 0.1 --graph_weight 0.8 --use_weight --attn_emb_dim 16 \
    --label_grad --runs 5 --device 3


python main.py --method augTrain --encoder geomix_1 \
    --dataset citeseer --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.1 --dropout 0.3  --hops 2 --alpha 0.7 --label_grad \
    --use_bn --runs 5 --device 1


python main.py --method augTrain --encoder geomix_2 \
    --dataset citeseer --lr 0.01 --num_layers 2 --hidden_channels 64 \
    --weight_decay 0.1 --dropout 0.3  --hops 2 --alpha 0.5 --label_grad \
    --use_bn --runs 5 --device 2

python main.py --method augTrain --encoder geomix_3 --dataset citeseer --lr 0.1 \
    --num_layers 2 --hidden_channels 16 --weight_decay 1e-2 --dropout 0.5 \
    --hops 2 --res_weight 0.5 --graph_weight 0.3 --use_weight  --attn_emb_dim 16 \
    --label_grad --use_bn --runs 5 --device 1


python main.py --method augTrain --encoder geomix_1 \
    --dataset pubmed --lr 0.3 --num_layers 2 --hidden_channels 16 \
    --weight_decay 0.001 --dropout 0.4  --hops 2 --alpha 0.1 --label_grad \
    --runs 5 --device 2


python main.py --method augTrain --encoder geomix_2 \
    --dataset pubmed --lr 0.3 --num_layers 2 --hidden_channels 16 \
    --weight_decay 5e-3 --dropout 0.6 --hops 2 --alpha 0.1 --label_grad \
    --no_feat_norm --runs 5 --device 2


python main.py --method augTrain --encoder geomix_3 --dataset pubmed --lr 0.3 \
    --num_layers 2 --hidden_channels 16 --weight_decay 0.0008 --dropout 0.3 \
    --hops 3 --res_weight 0.2 --graph_weight 0.5 --use_weight --attn_emb_dim 16 \
    --label_grad --no_feat_norm --runs 5 --device 1

python main.py --method augTrain --encoder geomix_1 \
    --dataset cs --lr 0.005 --num_layers 2 --hidden_channels 64 \
    --hops 1 --alpha 0.8 --weight_decay 5e-5 --dropout 0.5  \
    --device 1 --runs 5 --epochs 300


python main.py --method augTrain --encoder geomix_2 \
    --dataset cs --lr 0.005 --num_layers 2 --hidden_channels 64 \
    --hops 1 --alpha 0.8 --weight_decay 5e-5 --dropout 0.5  \
    --device 1 --runs 5 --epochs 300 


python main.py --method augTrain --encoder geomix_3 --dataset cs --lr 0.005 \
    --num_layers 2 --hidden_channels 64 --weight_decay 1e-4 --dropout 0.3 \
    --hops 1 --res_weight 0.7 --graph_weight 0.5 --use_weight --attn_emb_dim 16 \
    --runs 5 --device 1 --epochs 300


python main.py --method augTrain --encoder geomix_1 \
    --dataset physics --lr 0.005 --num_layers 2 --hidden_channels 64 \
    --hops 1 --alpha 0.8 --weight_decay 5e-5 --label_grad --dropout 0.3  \
    --device 2 --runs 5 --epochs 200


python main.py --method augTrain --encoder geomix_2 \
    --dataset physics --lr 0.005 --num_layers 2 --hidden_channels 64 \
    --hops 1 --alpha 0.8 --weight_decay 5e-5 --label_grad --dropout 0.3  \
    --device 2 --runs 5 --epochs 300


python main.py --method augTrain --encoder geomix_3 --dataset physics --lr 0.01 \
    --num_layers 2 --hidden_channels 64 --weight_decay 5e-4 --dropout 0.5 \
    --hops 1 --res_weight 0.8 --graph_weight 0.7 --use_weight --attn_emb_dim 16 \
    --label_grad --runs 5 --device 3 --epochs 150


