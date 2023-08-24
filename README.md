# DropNaE: Alleviating Irregularity for Large-scale Graph Representation Learning
An implementation of DropNaE: a novel method to alleviate the irregular neighbor distribution in large-scale graphs by conditionally dropping nodes and edges. Please follow this guideline to verify the performance of DropNaE. Since DropNaE is a once-for-all method performed offline, we have processed all large-scale graphs via DropNaE to generate the train_adj (adjacency matrix for training) files. Therefore, one can directly load the processed adj together with other data of one dataset to witness acceleration of training GNNs straightforwardly.
## Datasets
All datasets used in our papers are available for download:
* reddit
* amazon
* yelp
* ogbn-product 
One can download the datasets from [Google Drive](https://drive.google.com/drive/folders/1UM7WgCLvMX1ToMXcKn0lKG7DNkSwfNZE?usp=sharing).
**NOTE**: Datasets used in PGS-GNN are the same as those used in GraphSAINT. For example, one can just add `adj_train_N.npz` or `adj_train_E.npz` provided in ```PGS-GNN/data/dataset_name``` to the corresponding directory. 

The directory structure should be as below:
```
README.md
DropNaE/
Models/
└───GraphSAINT/ 
└─────data/
└───────Amazon/
└───────Reddit/
......
|   
└───Cluster-GCN/
└─────data/
└───────Amazon/
└───────Reddit/
......
└───PGS-GNN/
└─────data/
└───────Amazon/
└───────Reddit/
......
```
## Usage
To reproduce the results of our experiments, simply go into each model directory and run the prepared scripts. Please download the datasets and place them in correct locations as given in [Dataset](#Datasets).

Run the original version (baseline):
```
./Models/<model_name>/run_<dataset_name>.sh
```
Run the version with DropNaE utilized:
```
modify the run_<dataset_name>.sh by adding `--use_DropNaE N/E` mark, and then run:
./Models/<model_name>/run_<dataset_name>.sh
```
Run your own DropNaE:
```
python DropNaE.py [dataset] [drop_type] [model_name]
```
Args description:
- dataset : Datasets used in our paper include `reddit, yelp, amazon, ogbn-product`
- drop_type : We provide two methods `DropNaE-N` and `DropNaE-E`, which are denoted by `N` and `E`
- model_name : Our paper supports three models `GraphSAINT、PGS_GNN、Cluster_gcn`. One can use `saint, pg, clu` to select a model for experiments.
Example:
```
python DropNaE.py reddit E saint
```
Then, move the processed data to the corresponding directory.

## Experimental Devices
| Platform | Configuration |
|---|---
| CPU | Intel Xeon CPU E5-2650 v4 CPUs (dual 24-core) |
| GPU | NVIDIA Tesla V100 GPU (16 GB memory) |


## Dependencies
* python
* pytorch
* tensorflow
* numpy
* scipy
* scikit-learn
* pyyaml
* [metis](https://github.com/google-research/google-research/tree/master/cluster_gcn) = 5.1.0
* networkx = 1.11

## Acknowledgements
The proposed DropNaE is applied to recent state-of-the-art sampling-based GNNs to demonstrate the effectiveness of DropNaE in terms of efficiency and accuracy. We use the implementations of [Cluster-GCN](https://github.com/google-research/google-research/tree/master/cluster_gcn), [PGS-GNN](https://github.com/ZimpleX/gcn-ipdps19), and [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT) as backbones, and owe many thanks to the authors for making their code available. Moreover, we thank the authors of [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT) for offering download links to many datasets. 
