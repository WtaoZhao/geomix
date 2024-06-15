# GeoMix

Pytorch implementation of GeoMix: Towards Geometry-Aware Data Augmentation [1].

## Environment Set Up

Install the required packages according to requirements.txt.

Tip: Installing `torch_geometric` with a specific version number may not be easy. I think it is OK to install a version of `torch_geometric` different from the one I used, as long as it is compatible with your hardware.

## Dataset

For the Squirrel and Chameleon datasets, please download the data files from [here](https://github.com/yandex-research/heterophilous-graphs/tree/main/data). Save the Squirrel data file in the `./data/wiki_new/squirrel` folder, and save the Chameleon data file in the `./data/wiki_new/chameleon` folder.

For the Pileup dataset, please download the following files from this [link](https://zenodo.org/records/8015774) provided in this GitHub [repo](https://github.com/Graph-COM/StruRW): test_gg_PU10.root, test_gg_PU30.root, test_gg_PU50.root, test_qq_PU10.root and test_qq_PU30.root. Save all the files in the `./data/pileup` folder.

We provide our extracted features of images from Cifar10 and STL10 datasets in `./data`.

For other datasets, our scripts will automatically download data files.

## Usage Example

To run GeoMix on Cora dataset, switch to src/main_exp and run the following command:
```
# GeoMix-I
python main.py --method augTrain --encoder geomix_1 --dataset cora --lr 0.1 \
    --num_layers 2 --hidden_channels 16 --weight_decay 5e-4 --dropout 0.5 \
    --device 1 --alpha 0.1 --hops 3 --label_grad --runs 5 

# GeoMix-II
python main.py --method augTrain --encoder geomix_2 --dataset cora --lr 0.1 \
    --num_layers 2 --hidden_channels 16 --weight_decay 5e-4 --dropout 0.5 \
    --device 1 --hops 2 --alpha 0.1 --label_grad --runs 5

# GeoMix-III

python main.py --method augTrain --encoder geomix_3 --dataset cora --lr 0.1 \
    --num_layers 2 --hidden_channels 16 --weight_decay 5e-4 --dropout 0.5 \
    --hops 2 --res_weight 0.1 --graph_weight 0.8 --use_weight --attn_emb_dim 16 \
    --label_grad --runs 5 --device 3
```
Please refer to .sh scipts in `exp` folder in each subdirectory of `./src` for experiments on other datasets. Note that you may need to change GPU device ID by adjusting the value of the `--device` argument.

## References

[1] Wentao Zhao, Qitian Wu, Chenxiao Yang, and Junchi Yan. 2024. GeoMix: Towards Geometry-Aware Data Augmentation (KDD '24).


## Citation

If you find our code helpful, please consider citing our work
```
@inproceedings{zhao2023glow,
  title={GeoMix: Towards Geometry-Aware Data Augmentation},
  author={Zhao, Wentao and Wu, Qitian and Yang, Chenxiao and Yan, Junchi}
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2024}
}
```
