import torch
import numpy as np
from fizzbuzz import  FizzBuzz
from torch.nn import Linear, ReLU, Module, Dropout

INPUT_WIDTH=16

def fizzbuzz(x):
    if x % 15 == 0:
        return "FizzBuzz"
    elif x % 5 == 0:
        return "Buzz"
    elif x % 3 == 0:
        return "Fizz"
    else:
        return x


def encode_input(x: int):
    xs = np.binary_repr(x, width=INPUT_WIDTH)
    return np.array([int(i) for i in xs])


def decode_output(x, results: torch.Tensor):
    # outputs will be [num, Fizz, Buzz, FizzBuzz]
    return [x, "Fizz", "Buzz", "FizzBuzz"][results.tolist().index(max(results))]

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-r', '--range', type=int, default=10000)
    args = ap.parse_args()

    model = FizzBuzz(INPUT_WIDTH)
    model.load_state_dict(torch.load("fizzbuzz.pt"))
    model = model.cuda()
    model.eval()

    correct = 0
    incorrect = 0

    for i in range(1,args.range):
        
        out = model(torch.tensor(encode_input(i), dtype=torch.float32).to('cuda'))
        pred = decode_output(i, out)

        actual = fizzbuzz(i)

        if actual == pred:
            correct += 1
        else:
            incorrect += 1

        print(f'\rtesting {i} -----> {pred}', end='')
    
    print(f'\nCorrect percentage: {round(100*correct/(correct+incorrect), 2)}%')

