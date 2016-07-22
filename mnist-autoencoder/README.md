# MNIST (Denoising) Auto-Encoder Implementation in Chainer

Chainer code for training denoising auto-encoder with MNIST dataset.

## Usage:

```bash
$ python train_mnist_ae.py -a relu -u 1000 -b 1000 -e 20 -n 0.2
$ python dump_mnist_ae.py -m relu-1000units_batch1000-epoch20_noise0.2.model -a relu -u 1000 -n 50
$ open dump.png
```

### Example decoding results

<img src="https://dl.dropboxusercontent.com/u/1698760/github/relu-1000units_batch1000-epoch20_noise0.0.png">

(ReLU, 1000 hidden units, 0% noise)

<img src="https://dl.dropboxusercontent.com/u/1698760/github/sigmoid-1000units_batch1000-epoch20_noise0.2.png">

(sigmoid, 1000 hidden unis, 20% noise)
