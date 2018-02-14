#Define data
data_path = "data/anna.txt"

#Define hyperparameters
batch_size = 10         # Sequences per batch
num_steps = 50          # Number of sequence steps per batch
lstm_size = 128         # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.01    # Learning rate
keep_prob = 0.5         # Dropout keep probability
epochs = 20             # Number of epochs
