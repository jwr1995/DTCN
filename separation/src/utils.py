import numpy as np
from sklearn import cluster
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import euclidean_distances as dist

def normalize(mat, neg=False):
    min_value = np.min(mat)
    data = mat-min_value
    max_value = np.max(data)
    data = data/max_value
    if not neg:
        return data
    else:
        data = data*2-1
        return data

euclidean_matrix = lambda X : dist(X,X)

distance_triangle = lambda X : [list(row[:i]) for i, row in enumerate(euclidean_matrix(X))]


def channel_sort(X):
    # N x T
    dist_mat = euclidean_matrix(X)
    max_val = np.max(dist_mat)+1
    dist_mat[dist_mat==0.0] = max_val
    channel_order = np.where(dist_mat == dist_mat.min())[0] # starting channels
    subset_1 = dist_mat[channel_order[0]]

    subset_1[channel_order[1]] = max_val
    

    while len(channel_order) < X.shape[0]:
        min_arg_1 = np.argmin(subset_1)
        channel_order = np.insert(channel_order,0, min_arg_1)
        
        subset_1[min_arg_1] = max_val
        
       

    return channel_order

from UPGMApy.UPGMA import UPGMA


def count_clusters(no_clusters, clusters):
    print(type(tuple()),type(clusters[0]),print(clusters[1]))
    if type(tuple()) == type(clusters[0]):
        no_clusters+=1
        no_clusters=count_clusters(no_clusters,clusters[0])
    return no_clusters

def unpack_clusters(clusters,label_list=None,):
    if label_list == None:
        label_list = []
    
    for i in range(len(clusters)):
        if type(clusters[-i]) == type(int(0)):
            label_list.append(clusters[-i])
        else:
            label_list = unpack_clusters(clusters[-i],label_list)
    
    return label_list

def tuple_string_to_list(tuple_string):
    return [entry for entry in tuple_string.replace(")","").replace("(","").split(",")]

def tuple_string_to_enumerated_list(tuple_string):
    return [int(entry) for entry in tuple_string.replace(")","").replace("(","").split(",")]


def upgma_channel_sort(W):
    N = len(W) #.shape # out channels, in channels, kernel size
    W = W.squeeze()
    W = distance_triangle(W)
    channel_labels = [str(i) for i in range(N)]
    assert len(W) == len(channel_labels)
    upgma_clusters = UPGMA(W,channel_labels)
    channel_order = tuple_string_to_enumerated_list(upgma_clusters)
    return channel_order


if __name__ == '__main__':
    from UPGMApy.UPGMA import alpha_labels
    import sys
    # sys.setrecursionlimit(10000)

    # M_labels = alpha_labels("A", "G")   #A through G
    # M = [
    #     [],                         #A
    #     [19],                       #B
    #     [27, 31],                   #C
    #     [8, 18, 26],                #D
    #     [33, 36, 41, 31],           #E
    #     [18, 1, 32, 17, 35],        #F
    #     [13, 13, 29, 14, 28, 12]    #G
    #     ]

    # clusters = UPGMA(M, M_labels) # should output: '((((A,D),((B,F),G)),C),E)'
    # print(clusters)
    # l_list = tuple_string_to_list(clusters)
    # print(l_list)

    W = np.random.rand(512,1,16)
    channel_order=upgma_channel_sort(W)
    print(channel_order)
    print(channel_sort(W.squeeze()))


    # #####################################    
    # K=10
    # N=10
    # X = np.random.rand(K,N)
    # X = channel_sort(X)
    # print(X)
