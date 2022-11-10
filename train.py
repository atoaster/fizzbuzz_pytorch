import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from fizzbuzz import FizzBuzz
import numpy as np

INPUT_WIDTH=16

def encode_input(x: int):
    xs = np.binary_repr(x, width=INPUT_WIDTH)
    return np.array([int(i) for i in xs])

def fizzbuzz(x):
    if x % 15 == 0:
        return "FizzBuzz"
    elif x % 5 == 0:
        return "Buzz"
    elif x % 3 == 0:
        return "Fizz"
    else:
        return x


def decode_output(x, results: list):
    # outputs will be [num, Fizz, Buzz, FizzBuzz]
    return [x, "Fizz", "Buzz", "FizzBuzz"][list.index(max(results))]


def fizz2vec(result):
    if result == "Fizz":
        return [0, 1, 0, 0]
    elif result == "Buzz":
        return [0, 0, 1, 0]
    elif result == "FizzBuzz":
        return [0, 0, 0, 1]
    else:
        return [1, 0, 0, 0]




def make_training_set(training_size=1000):
    X = []
    Y = []

    for i in range(1, training_size):
        X.append(encode_input(i))

        actual_result = fizzbuzz(i)
        Y.append(fizz2vec(actual_result))

    X = np.array(X)
    Y = np.array(Y)
    return DataLoader(
        TensorDataset(
            torch.tensor(X, device="cuda", dtype=torch.float32),
            torch.tensor(Y, device="cuda", dtype=torch.float32),
        ), batch_size=training_size
    )

model = FizzBuzz(INPUT_WIDTH)
model = model.cuda()
model.train()

loss_func = CrossEntropyLoss().cuda()

opt = Adam(model.parameters(), lr=5e-3)

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--epochs", default=1000, type=int)
    ap.add_argument('-t', '--training_size', default=1024, type=int)
    args = ap.parse_args()

    print('making training dataset')
    dataset = make_training_set(args.training_size)

    print()
    for epoch in range(args.epochs):
        epoch_loss = 0
        for batch_index, batch in enumerate(dataset):
            x, y = batch

            # loss = update(x, y)
            yt = model(x)
            loss = loss_func(yt, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()

        if epoch%50 == 0:
            print(f"Epoch={epoch}, loss={epoch_loss:4.4f}")


    print()
    torch.save(model.state_dict(), "fizzbuzz.pt")
