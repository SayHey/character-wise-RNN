from data import *

vocab, anna = load_data("data/anna.txt")

batches = get_batches(anna, 10, 50)
x, y = next(batches)

print('x\n', x[:10, :10])
print('\ny\n', y[:10, :10])