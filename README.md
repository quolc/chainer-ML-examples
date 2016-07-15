# Simple Perceptron for Linear Separation in Chainer

Chainer code for training 2-input 2-output simple perceptron.
gen_data.py generates training data for '<' function.

## Usage:

```bash
$ python gen_data.py 10000 > train.txt
$ python train_linear.py < train.txt
$ python evaluate_linear.py
```

```bash
100 200
-> a < b
1 0.5
-> a > b
```

