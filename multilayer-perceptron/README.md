# Three-layer Perceptron in Chainer

Chainer code for training 2-input N-hidden-layer 2-output perceptron with back propagation.
gen_data.py generates training data of swirling pattern.

## Usage:

```bash
$ python gen_data.py 10000 > train.txt
$ python train_three.py < train.txt
$ python visualize_three.py
```

### Example classification result (6 Hidden Units)

<img src="https://dl.dropboxusercontent.com/u/1698760/github/three_checker_6units.png" />
