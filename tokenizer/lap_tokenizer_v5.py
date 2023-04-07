# Purpose of having tokenizer: to have a cheaper representation of graph

import h5py
import numpy as np
import torch

max_edges = 1000

def lap_eigen(A, dp):  # expecting a sparse matrix, so it reduce the size
    '''    
    Revised Version 2
    input:  A - adjacency matrix (NxN), dp - scalar dimension of desire (recommend 3). 
        Consider as "how much info do you want to keep during the tokenizing process"
        tokenGT use dp=3
    output: lap_eigvec - laplacian eigen vectors of dimensions N x dp
        intuitively, it is a way to capture information of edges on each node (consider it as attention to nodes)
    '''
    
    # Compute Matrix Dimension:
    N = A.shape[0]

    # Compute Degree Matrix:
    D = torch.diag(torch.sum(A, axis=1))

    # Compute normalized Laplacian matrix L
    #D_sqrt_inv = np.sqrt(np.linalg.inv(D))
    D_sqrt_inv = torch.sqrt(torch.linalg.pinv(D))
    L = torch.eye(A.shape[0]).to(A.device) - torch.matmul(torch.matmul(D_sqrt_inv, A), D_sqrt_inv)

    # Perform eigendecomposition of L
    eigvals, eigvecs = torch.linalg.eig(L)
    sorted_indices = torch.real(eigvals).argsort()
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    # Return this
    if len(sorted_indices) >= dp:
        lap_eigvec = eigvecs[:, 1:dp + 1]
    elif len(sorted_indices) < dp:
        lap_eigvec = np.zeros((N, dp))
        lap_eigvec[:, :len(sorted_indices)] = eigvecs

    return lap_eigvec


def token_construction(A, Xv, dp=3):
    '''
    input: A - adjacency matrix (NxN), Xv - nodes (Nx1)
    output: X - token, should be in dimension (N + M) x (1 + 2dp + 4)
    approach: laplacian eigenvector implementation
    '''
    
    # Compute Matrix Dimension:
    N = A.shape[0]

    # Find all the edges:
    node_pairs = [(i, j) for i in range(N) for j in range(N) if A[i, j] != 0]
    #print(len(node_pairs))
    #print(node_pairs)

    # for every edge, create an augmented feature matrix Xe (M x 1, M is amount of edges)
    Xe = []
    for pair in node_pairs:
        row = [A[pair[0], pair[1]]]
        Xe.append(row)
    Xe = torch.tensor(Xe).to(A.device)

    # Construct node identifier matrix P
    #print(A)
    P = lap_eigen(A, dp)  # laplace value

    # Augment P's and E's into X
    # Ev/Ee: type identifiers for nodes and edges
    Ev = []  
    '''
    identifier: N x 4,
        for indice 0 and 1, [0, 1] -> Node and [1, 0] -> Edge 
        for indice 2 and 3, 
            nodes always have [-1, -1], 
            Edge is [u, v], where u is the index (identifier) of node 1, v is the index of node 2
    '''
    
    for i in range(0, N, 1):
        Ev.append([1, 0, -1, -1])
    Ev = torch.tensor(Ev).to(A.device)
    new_Xv = torch.hstack((Xv, P, P, Ev))  # N x (1 + 3 + 3 + 4)

    new_Xe = []
    for i in range(0, Xe.shape[0], 1):
        u, v = node_pairs[i]
        new_Xe.append(torch.hstack((Xe[i], P[u], P[v], torch.tensor([0, 1, u, v]).to(A.device))))
    new_Xe = torch.stack(new_Xe)

    # Concat Xv and Xe vertically
    X = torch.vstack((new_Xv, new_Xe))

    padding = max_edges - len(node_pairs)

    if padding > 0:  # zoom padding
        # padded_X = torch.pad(X, [(0, padding), (0, 0)], mode='constant')
        padded_X = torch.nn.functional.pad(X, pad=(0, 0, padding, 0), mode='constant', value=0)
        #print(padded_X.shape)
    else:  # cropping out the additional edge. If the max_edge is set correctly, this shouldn't happen
        print("hitting else: Max_edge is not long enough")
        padded_X = X[:(max_edges + N), :]
        

    # print("padded_X.shape", padded_X.shape)
    # print("new_Xv.shape, new_Xe.shape", new_Xv.shape, new_Xe.shape)

    return padded_X

def token_construction_batch(As, Xvs):
    N = As.shape[0]
    everything = []
    for i in range(0, N, 1):
        # everything.append(np.real(token_construction(As[i].cpu().numpy(), Xvs[i].cpu().numpy())).astype(np.float32))
        everything.append(torch.real(token_construction(As[i], Xvs[i])).to(dtype=torch.float32))
    return torch.stack(everything)

# Test 1: test the tokenizer for a random graph
def test_tokenizer_for_a_random_graph():
    Atp = np.random.randint(2, size=(5, 5))
    Ntp = np.random.randint(10, size=(5, 3))
    np.fill_diagonal(Atp, 0)
    print(Atp)
    print(token_construction(Atp, Ntp))
    return True



# Test 2: test the tokenizer for a random graph in validation set of deepnet1m
def test_tokenizer_for_a_graph_in_deepnet1m_val():
    filename = "../data/deepnets1m_eval.hdf5"
    with h5py.File(filename, "r") as f:

        # key 4 is validation set
        a_group_key = list(f.keys())[4]
        group = f[a_group_key]
        # select a random piece of data (index 42) from the validation set
        sample_key = list(group.keys())[42]
        # grab the adjacency matrix
        sample_data_adj = group[sample_key]['adj']
        sample_data_nodes = group[sample_key]['nodes']

        # Take a look at the data
        print(sample_data_adj)
        print(sample_data_adj[()])
        print(sample_data_nodes)
        print(sample_data_nodes[()])

        # Take a look at the token
        print(token_construction(sample_data_adj[()], sample_data_nodes[()]))
    return True

if __name__ == "__main__":
    test_tokenizer_for_a_random_graph()  # Test 1
    test_tokenizer_for_a_graph_in_deepnet1m_val()  # Test 2