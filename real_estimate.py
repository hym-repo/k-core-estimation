import bisect
import math
import os
import random
import time


import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

APPLY_DEGREE_CUTOFF = True

def k_seq_estimate(seq):
    deg_,count_ = np.unique(seq,return_counts=True)
    deg_pos = {j: i for i, j in enumerate(deg_)}
    pmf = count_ / np.sum(count_)
    cdf = np.cumsum(pmf)                     
    ecdf = np.cumsum(pmf * deg_)          
    c = ecdf[-1]
    m = np.sum(seq)
    N = np.sum(count_)


    estimates = np.zeros(len(deg_))
    for id_,w in enumerate(deg_):
        if id_ == 0:
            estimates[id_] = deg_pos[w]
            continue
        
        wc = math.ceil(m / w)
        wc_pos = (bisect.bisect_left(deg_, wc) if wc<deg_[-1] else len(deg_))
                
        if w < wc:
            estimates[id_] = w / c * (ecdf[wc_pos-1]-ecdf[deg_pos[w]-1]) + N * (cdf[-1]-cdf[wc_pos-1])
        else:
            estimates[id_] = N * (cdf[-1]-cdf[deg_pos[w]-1])

            
    kh = np.max(estimates)
    wh = np.argmax(np.array(estimates))


    return kh,wh

def edgelist_to_adj_csr(arr,N):
    row = arr[:, 0]
    col = arr[:, 1]
    rows = np.concatenate((row,col))
    cols = np.concatenate((col,row))
    data = np.ones(len(rows))
    A_csr = csr_matrix((data, (rows,cols)), shape=(N, N))

    return A_csr


if __name__ == "__main__":
    root_path = 'Data'
    current_path = [root_path]    
    random.seed(time.time())

    for dataset in os.listdir(root_path):
        current_path.append(dataset)
        k_estimates = np.zeros(2)
        
        for file in os.listdir('/'.join(current_path)):
            current_path.append(file)
            edges = np.loadtxt('/'.join(current_path),delimiter=',',dtype='int')            
            N = np.max(edges.flatten())+1
                        
            A_csr = edgelist_to_adj_csr(edges,N)            
            degrees = np.array(np.sum(A_csr,axis=1),dtype=int).flatten()
            if APPLY_DEGREE_CUTOFF:
                volume = np.sum(degrees)
                degrees_cutoff = np.zeros(len(degrees))
                for w_id,w in enumerate(degrees):
                    w_links = w * degrees / volume
                    w_e = np.sum(w_links)
                    w_links[w_links>1] = 1
                    w_c = np.sum(w_links)
                    degrees_cutoff[w_id] = round(w_c)
                degrees = degrees_cutoff
            try:                
                start_ = time.perf_counter()
                kh_,wh_ = k_seq_estimate(degrees)
                end_ = time.perf_counter()                
                k_estimates=[kh_,end_-start_]
                k_savepath = ['./output',f'realworld_E',dataset,'k',file]
                if not os.path.exists('/'.join(k_savepath[:-1])):
                    os.makedirs('/'.join(k_savepath[:-1]))            
                np.savetxt('/'.join(k_savepath),k_estimates,delimiter=',',fmt='%.8f')            
                current_path.pop()
            except:
                print(current_path)
                current_path.pop()
                continue
        current_path.pop()
    


