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
saint_arg_dict = {'reddit':[],'yelp':[],'amazon':[],'ogbn-product':[]}
pg_arg_dict = {'reddit':[],'yelp':[],'amazon':[],'ogbn-product':[]}
clu_arg_dict = {'reddit':[],'yelp':[],'amazon':[],'ogbn-product':[]}

if model == 'saint':
    p_s, p_d = saint_arg_dict[dataset_str]
    DropNaE(p_s, p_d, prefix=prefix, Drop_type=drop_type)
elif model == 'pg':
    p_s, p_d = pg_arg_dict[dataset_str]
    DropNaE_pg(p_s, p_d, prefix=prefix, Drop_type=drop_type)
elif model == 'clu':
    p_s, p_d = clu_arg_dict[dataset_str]
    DropNaE_clu(p_s, p_d, dataset_str=dataset_str, Drop_type=drop_type)