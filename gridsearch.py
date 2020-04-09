import numpy as np
import train


def main():
    param_set = [(q, i) for q in np.arange(0.01, 0.11, 0.01) for i in np.arange(0.02, 0.11, 0.02)]
    for param in param_set:
        print(param)
        train.train(param[0], param[1])


if __name__ == '__main__':
    main()
