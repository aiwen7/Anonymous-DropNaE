import torch
import numpy as np
import scipy.sparse as sp
from random import sample
import json
import sys
import networkx as nx
import os
from networkx.readwrite import json_graph
from utils import *

dataset_str, drop_type, model = sys.argv[1:]
prefix = 'data/' + dataset_str
saint_arg_dict = {'reddit':[0.1,0.2],'yelp':[0.15,0.05],'amazon':[0.3,0.05],'ogbn-product':[0.7,0.1]}
pg_arg_dict = {'reddit':[0.3,0.1],'yelp':[0.3,0.05],'amazon':[0.3,0.05],'ogbn-product':[0.3,0.3]}
clu_arg_dict = {'reddit':[0.3,0.2],'yelp':[0.3,0.3],'amazon':[0.3,0.05],'ogbn-product':[0.7,0.1]}

if model == 'saint':
    p_s, p_d = saint_arg_dict[dataset_str]
    DropNaE(p_s, p_d, prefix=prefix, Drop_type=drop_type)
elif model == 'pg':
    p_s, p_d = pg_arg_dict[dataset_str]
    DropNaE_pg(p_s, p_d, prefix=prefix, Drop_type=drop_type)
elif model == 'clu':
    p_s, p_d = clu_arg_dict[dataset_str]
    DropNaE_clu(p_s, p_d, dataset_str=dataset_str, Drop_type=drop_type)