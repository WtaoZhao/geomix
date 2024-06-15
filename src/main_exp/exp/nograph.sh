python main.py --method augTrain --encoder geomix_1 --dataset stl10 --label_num_per_class 10  \
    --valid_num 4000 --lr 0.005 --num_layers 2 --hidden_channels 32 \
    --use_bn --hops 2 --alpha 0 --label_grad --weight_decay 1e-3 --dropout 0.5  \
    --device 0 --runs 5 --epochs 300 

python main.py --method augTrain --encoder geomix_2 --dataset stl10 --label_num_per_class 10  \
    --valid_num 4000 --lr 0.005 --num_layers 2 --hidden_channels 32 \
    --use_bn --hops 2 --alpha 0.1 --label_grad --weight_decay 1e-5 --dropout 0.5  \
    --device 0 --runs 5 --epochs 300 

python main.py --method augTrain --encoder geomix_3 --dataset stl10 --label_num_per_class 10  \
    --valid_num 4000 --lr 0.005 --num_layers 2 --hidden_channels 32 \
    --use_bn  --label_grad --weight_decay 1e-5 --dropout 0.5  \
    --hops 2 --res_weight 0.8 --graph_weight 0.8 --use_weight \
    --device 0 --runs 5 --epochs 300 

python main.py --method augTrain --encoder geomix_1 --dataset cifar10 --label_num_per_class 10  \
    --valid_num 4000 --lr 0.001 --num_layers 2 --hidden_channels 32 \
    --use_bn --hops 2 --alpha 0 --label_grad --weight_decay 1e-2 --dropout 0.5  \
    --device 0 --runs 5 --epochs 300 

python main.py --method augTrain --encoder geomix_2 --dataset cifar10 --label_num_per_class 10  \
    --valid_num 4000 --lr 0.005 --num_layers 2 --hidden_channels 32 \
    --use_bn --hops 1 --alpha 0.8 --label_grad --weight_decay 1e-4 --dropout 0.4  \
    --device 0 --runs 5 --epochs 300 

python main.py --method augTrain --encoder geomix_3 --dataset cifar10 --label_num_per_class 10  \
    --valid_num 4000 --lr 0.005 --num_layers 2 --hidden_channels 32 \
    --use_bn --label_grad --weight_decay 1e-4 --dropout 0.4  \
    --hops 1 --res_weight 0.8 --use_weight --graph_weight 0.8 \
    --device 0 --runs 5 --epochs 300     

python main.py --method augTrain --encoder geomix_1 --dataset 20news --label_num_per_class 100  \
    --valid_num 2000 --lr 0.005 --num_layers 2 --hidden_channels 128 \
    --use_bn --hops 1 --alpha 0 --label_grad --weight_decay 5e-3 --dropout 0.6  \
    --device 0 --runs 5 --epochs 300 

python main.py --method augTrain --encoder geomix_2 --dataset 20news --label_num_per_class 100  \
    --valid_num 2000 --lr 0.005 --num_layers 2 --hidden_channels 128 \
    --use_bn --hops 2 --alpha 0.1 --label_grad --weight_decay 1e-3 --dropout 0.5  \
    --device 0 --runs 5 --epochs 300

python main.py --method augTrain --encoder geomix_3 --dataset 20news --label_num_per_class 100  \
    --valid_num 2000 --lr 0.005 --num_layers 2 --hidden_channels 128 \
    --use_bn --label_grad --weight_decay 1e-2 --dropout 0.5  \
    --hops 2 --res_weight 0.1 --use_weight --graph_weight 0.3 \
    --device 0 --runs 5 --epochs 300 



