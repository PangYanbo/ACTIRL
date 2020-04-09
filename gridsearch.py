import numpy as np
import train


def main():
<<<<<<< HEAD
    param_set = [(q, i) for q in np.arange(0.01, 0.11, 0.01) for i in np.arange(0.01, 0.11, 0.01)]
=======
    param_set = [(q, i) for q in np.arange(0.01, 0.11, 0.01) for i in np.arange(0.02, 0.11, 0.02)]
>>>>>>> 34eac65aa5c0a2b23c90c27fdc76b308af8af34d
    for param in param_set:
        print(param)
        train.train(param[0], param[1])


if __name__ == '__main__':
    main()
