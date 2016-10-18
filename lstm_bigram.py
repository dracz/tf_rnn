from data_io import read_zipped_text
import tensorflow as tf

filename = "/Users/dracz/data/text8.zip"

text = read_zipped_text(filename)
train_text, valid_text = split_text(text, 1000)
train_size, valid_size = len(train_text), len(valid_text)

print('{:>12,} total chars: "{}"'.format(len(text), text[:64]))
print('{:>12,} train chars: "{}"'.format(train_size, train_text[:64]))
print('{:>12,} valid chars: "{}"'.format(valid_size, valid_text[:64]))


