import h5py
import numpy as np


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

'''
# Test 1: Reconstruct token for a random graph
from lap_tokenizer_v3 import *
Atp = np.random.randint(2, size=(4, 4))
Ntp = np.random.randint(10, size=(4, 3))
np.fill_diagonal(Atp, 0)
print(Atp, Ntp)
print(token_construction(Atp, Ntp).real)
print(token_destruction(token_construction(Atp, Ntp)))
'''

# Test 1: Reconstruct token for a random graph
def reconstruct_token_for_a_random_graph():  
    from lap_tokenizer_v3 import token_construction
    Atp = np.random.randint(2, size=(4, 4))
    Ntp = np.random.randint(10, size=(4, 3))
    np.fill_diagonal(Atp, 0)
    print(Atp, Ntp)
    tokenized_output = token_construction(Atp, Ntp)
    print(tokenized_output.real)
    print(token_destruction(tokenized_output))
    return True

if __name__ == "__main__":
    reconstruct_token_for_a_random_graph()  # Test 1
