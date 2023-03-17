import h5py
import numpy as np
from numpy.testing import assert_almost_equal


def token_destruction(X):
    '''    
    input: X - token, should be in dimension (N + M) x (3 + 2dp + 4)
    output: A - adjacency matrix (NxN), Xv - nodes (Nx3)
    approach: reverse laplacian eigenvector implementation
    '''
    N = 0
    for i in range(0, X.shape[0], 1):
        snip = X[i, -4:-2].real.astype(int)
        if snip[0] == 1 and snip[1] == 0:
            N += 1
            continue
        break

    # Obtain nodes back
    Xv = X[:N, :3].real

    # Obtain the adjacency matrix back
    A = np.zeros((N, N))

    cnt = 0
    for pair in X[N:, -2:].real.astype(int):
        #print(pair)
        A[pair[0], pair[1]] = X[N + cnt, 0].real.astype(int)
        cnt += 1

    return A, Xv


# Test 1: Reconstruct token for a random graph
def reconstruct_token_for_a_random_graph():  
    from lap_tokenizer_v3 import token_construction
    Atp = np.random.randint(2, size=(4, 4))
    Ntp = np.random.randint(10, size=(4, 3))
    np.fill_diagonal(Atp, 0)
    tokenized_output = token_construction(Atp, Ntp)
    reconstructed_graph = token_destruction(tokenized_output)
    
    # print(Atp, Ntp)
    # print(tokenized_output.real)
    # print(reconstructed_graph)
    
    assert_almost_equal(reconstructed_graph[0], Atp)
    assert_almost_equal(reconstructed_graph[1], Ntp)
    return True


def reconstruct_token_for_a_graph_in_deepnet1m_val():
    from lap_tokenizer_v3 import token_construction
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

        # tokenize and de-tokenize
        tokenized_output = token_construction(sample_data_adj[()], sample_data_nodes[()])
        reconstructed_graph = token_destruction(tokenized_output)
        
        # # Take a look at the input data
        # print(sample_data_adj)
        # print(sample_data_adj[()])
        # print(sample_data_nodes)
        # print(sample_data_nodes[()])
        # # Take a look at the tokenized_output
        # print(tokenized_output)
        # # Take a look at the de-tokenized_output
        # print(reconstructed_graph)
        
        assert_almost_equal(reconstructed_graph[0], sample_data_adj[()])
        assert_almost_equal(reconstructed_graph[1], sample_data_nodes[()])
    return True

if __name__ == "__main__":
    reconstruct_token_for_a_random_graph()  # Test 1
    reconstruct_token_for_a_graph_in_deepnet1m_val()  # Test 2
    print('all tests passed')