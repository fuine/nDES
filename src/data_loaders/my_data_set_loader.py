from math import ceil


class MyDatasetLoader:
    def __init__(self, x_train, y_train, batch_size):
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.num_batches = int(ceil(x_train.shape[0] / batch_size))
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        i = self.i
        idx = i * self.batch_size
        self.i += 1
        if i < self.num_batches - 1:
            return i, (
                self.x_train[idx: idx + self.batch_size],
                self.y_train[idx: idx + self.batch_size],
            )
        elif i == self.num_batches - 1:
            output = i, (self.x_train[idx:], self.y_train[idx:])
            self.i = 0
            return output

    def __len__(self):
        return self.num_batches