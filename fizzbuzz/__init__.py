from torch.nn import Linear, Dropout, ReLU, Module

class FizzBuzz(Module):
    def __init__(self, input_width) -> None:
        super().__init__()
        self.f1 = Linear(input_width, 256)
        self.d1 = Dropout()
        self.f2 = Linear(256, 512)
        self.relu = ReLU()
        self.out = Linear(512, 4)

    def forward(self, x):
        x = self.f1(x)
        x = self.d1(x)
        x = self.f2(x)
        x = self.relu(x)
        return self.out(x)
