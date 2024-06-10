#!/usr/bin/env python3
import os
import sys
import time
import tqdm
import pprint
import logging
import argparse

import numpy as np

from cuda.cudart import cudaGetDeviceProperties, cudaDeviceSynchronize

from nanodb import cudaVectorIndex, DistanceMetrics
from nanodb.utils import LogFormatter, tqdm_redirect_stdout


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-d', '--dim', type=int, default=768, help='the dimensionality of the embedding vectors') 
parser.add_argument('-n', '--num-vectors', type=int, default=4096, help='the number of vectors to add to the index')
parser.add_argument('-k', type=int, default=32, help='the number of search results to return per query')

parser.add_argument('--reserve', type=int, default=1024, help="the memory to reserve for the database in MB")
parser.add_argument('--dtype', type=str, default='float16', choices=['float32', 'float16'], help="datatype of the vectors")
parser.add_argument('--metric', type=str, default='cosine', choices=DistanceMetrics, help="the distance metric to use during search")
parser.add_argument('--max-search-queries', type=int, default=1, help="the maximum number of searches performed in one batch")

parser.add_argument('--seed', type=int, default=1234, help='change the random seed used')
parser.add_argument('--num-queries', type=int, default=1, help='the number of searches to run')

args = parser.parse_args()
LogFormatter.config(level='debug')

print(args)

np.random.seed(args.seed)

_, device_props = cudaGetDeviceProperties(0)
logging.info(f"cuda device:  {device_props.name}")

index = cudaVectorIndex(
    args.dim, 
    dtype=args.dtype, 
    reserve=args.reserve*1024*1024, 
    metric=args.metric, 
    max_search_queries=args.max_search_queries
)

print('-- generating random test vectors')
vectors = np.random.random((args.num_vectors, args.dim)).astype(index.dtype)
#vectors[:, 0] += np.arange(args.num_vectors) / 1000.
queries = np.random.random((args.num_queries, args.dim)).astype(index.dtype)
#queries[:, 0] += np.arange(args.num_queries) / 1000.

print('-- vectors', vectors.shape, vectors.dtype)
print('-- queries', queries.shape, queries.dtype)

for n in range(args.num_vectors):
    index.add(vectors[n])

print(f"-- added {index.shape} vectors")
print(f"-- validating index")

index.validate()

for n in tqdm.tqdm(range(args.num_vectors), file=sys.stdout):
    with tqdm_redirect_stdout():
        indexes, distances = index.search(vectors[n], k=args.k)
        if indexes[0] != n:
            print(f"incorrect index[{n}]\n", indexes, "\n", distances)
            assert(indexes[0] == n)
    
print(f"-- searching {queries.shape} vectors (metric={args.metric}, k={args.k})")
time_begin=time.perf_counter()
#for i in range(3):
indexes, distances = index.search(queries, k=args.k)
time_end=time.perf_counter()

print('\n', indexes)
print('\n', distances)
print(f"\n-- search time for n={args.num_vectors} d={args.dim} k={args.k}  {(time_end-time_begin)*1000} ms")

"""
for m in range(args.num_queries):
    search = index.search(xq[m], metric=args.metric)
    #print(search.shape)
    #print(search)
    
for m in range(args.num_vectors):
    search = index.search(xb[m], metric=args.metric)
    assert(search[0] == m)
    #print(search)
    #print(search.shape)
    #
"""
