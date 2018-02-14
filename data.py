import numpy as np

def load_data(file_path):
    with open(file_path, 'r') as f:
        text=f.read()
    vocab = sorted(set(text))
    vocab_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

    return vocab_to_int, int_to_vocab, vocab, encoded

def get_batches(arr, batch_size, n_steps):
    '''Create a generator that returns batches of size
       batch_size x n_steps from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    # Get the number of characters per batch and number of batches we can make
    characters_per_batch = batch_size * n_steps
    n_batches = int(len(arr) / characters_per_batch)
    
    # Keep only enough characters to make full batches
    size = characters_per_batch * n_batches
    arr = arr[:size]
    
    # Reshape into batch_size rows
    shape = (batch_size, -1)
    arr = arr.reshape(shape)
    
    for n in range(0, arr.shape[1], n_steps):
        # The features
        x = arr[:,n:n+n_steps]
        # The targets, shifted by one
        y = np.zeros_like(x)
        y[:,:-1], y[:,-1] = x[:, 1:], x[:,0]
        yield x, y