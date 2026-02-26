import os
import random
import time
import math

import networkx as nx
import numpy as np

from scipy.sparse import csr_matrix
from scipy.special import comb
from scipy.special import gammaln

def k_theory(degrees,k_ub,max_iter=1000,tol=1e-2):
    deg_,count_ = np.unique(degrees,return_counts=True)
    imax = round(deg_[-1])
    P = np.zeros(imax + 1)
    count_norm = count_ / np.sum(count_)
    for i,j in zip(deg_,count_norm):
        P[round(i)] = j
    z1 = sum([i * p for i,p in enumerate(P)])
    
    k_bound = [0,k_ub]
    
    while True:
        k_current = np.sum(k_bound) // 2 
        R_iter = 1e-5
        for t in range(max_iter):
            R_new = 1e-5
            for n in range(0, k_current - 1):  # n = 0 to k-2
                inner = 0.0
                for i in range(n, imax):
                    if P[i + 1] == 0:
                        continue
                    log_comb = (    
                                gammaln(i + 1)
                                - gammaln(n + 1)
                                - gammaln(i - n + 1)
                                )

                    log_term = (
                                np.log(i + 1)
                                + np.log(P[i + 1])
                                - np.log(z1)
                                + log_comb
                                + (i - n) * np.log(R_iter)
                                + n * np.log(1 - R_iter)
                                )
                    inner += np.exp(log_term)
                    # inner += (i + 1) * P[i + 1] / z1 * comb(i, n) * (R_iter ** (i - n)) * ((1 - R_iter) ** n)
                    if not np.isfinite(inner):
                        pass
                R_new += inner

            if abs(R_new-R_iter)<tol:
                break
            if t+1 == max_iter:
                pass
            R_iter =  np.clip(R_new, 1e-5, 1 - 1e-5)
            
        
        if k_bound[1]-k_bound[0] == 1:
            k_theory = k_bound[0]
            break
        
        
        if abs(R_iter-1)<tol:
            k_bound[1] = k_current
        else:
            k_bound[0] = k_current
            
    return k_theory    


def k_prun(A_csr):
    w_seq = np.sum(A_csr,axis=1)    
    k_ = 1
    while(len(w_seq)>0):
        if w_seq.min() > k_:
            k_ = w_seq.min()

        mask = np.ones(A_csr.shape[0], dtype=bool)
        mask_id = np.where(w_seq <= k_)[0]
        mask[mask_id] = False

        
        A_csr = A_csr[mask,:]
        A_csc = A_csr.tocsc()
        A_csc = A_csc[:,mask]
        A_csr = A_csc.tocsr()
        w_seq = np.sum(A_csr,axis=1)
    
    return k_

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
        try:
            for file in os.listdir('/'.join(current_path)):
                current_path.append(file)
                edges = np.loadtxt('/'.join(current_path),delimiter=',',dtype='int')            
                N = np.max(edges.flatten())+1

                A_csr = edgelist_to_adj_csr(edges,N)            
                degrees = np.array(np.sum(A_csr,axis=1),dtype=int).flatten()
                
                start_construct = time.perf_counter()
                G_rand = nx.expected_degree_graph(degrees,seed=random.Random(time.time()),selfloops=False)
                edges_rand = np.array(list(G_rand.edges()))
                A_csr_rand = edgelist_to_adj_csr(edges_rand, N)
                degree_seq = np.array(np.sum(A_csr_rand,axis=1),dtype=int).flatten()
                end_construct = time.perf_counter()

                start_prun = time.perf_counter()
                kc = k_prun(A_csr_rand)
                end_prun = time.perf_counter()
                
                
                start_R = time.perf_counter()
                kt = k_theory(degree_seq,np.max(degree_seq))
                end_R = time.perf_counter()
                    
                k_results = [kc,kt,end_prun-start_prun,end_R-start_R,end_construct-start_construct]
                
                k_savepath = ['./output',f'realworld_R',dataset,'k',file]
                if not os.path.exists('/'.join(k_savepath[:-1])):
                    os.makedirs('/'.join(k_savepath[:-1]))            
                np.savetxt('/'.join(k_savepath),k_results,delimiter=',',fmt='%.8f')            
                    
                current_path.pop()
        except:
            print(current_path)
        current_path.pop()
            