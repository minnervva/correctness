# PyTorch Tests

This directory holds all the PyTorch operation benchmark sweeps and the training and inference non-determinism tests.

## Repo organization

Please note the repository is organized as follows, with the operator benchmarks presented in the paper in `test-suite/paper-results` and the full sweep in the parent `test-suite`
```
test-suite/
│ └── data/
│ ├── benchmark and plotting scripts
├── paper-results/
  └── data
  ├── benchmark and plotting scripts
```


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the relevant packages. Please ensure you are in a virtual environment using conda, python venv etc. It is ideal to have a GPU to run these experiements, however this can be done on the CPU

```bash
pip install -r requirements.txt
```

## Usage

```sh
# For the full operator sweep, results will be stashed in csvs in `./data`
python scaffold.py

# For the train and inference test, figures will be plotted and saved in `test-suite`
python train.py

# To analyze data, please refer to operations_analysis.ipynb

# regenerate paper results, for the selected index_add and scatter_reduce operations. Data is stored in `paper-results/data`

python /paper-results/scaffold.py

# To analyze data, please refer to generate_operation_figures.ipynb
```

Note, to add your own sets of hyperparameters to test for non-determinism, please head to the relevat scaffold.py file and edit the relevant operation hyperparameters and global variables in the main body

```python

if __name__ == "__main__":

    deterministic = False
    device = gpu
    iterations = 1000
    
    conv2d_dimensions = {
        "batch_size": list([1, 2]),
        "dim": list([(3, 256, 256), (3, 512, 512), (3, 1024, 1024)]),
        "kernel_size": list([(3, 3), (5, 5), (7, 7)]),
        "stride": list([1, 3, 5]),
        "padding": list([0, 1]),
        "dilation": list([1, 2, 5]),
        "groups": list([1, 2, 3]),
        "dtype": list([torch.float32]), # TODO: check why relative error is NaN with fp16
    }
   
    print("=========================== Conv2d benchmark ======================================") 
    benchmark(
        op = nn.Conv2d,
        weight_dist = nn.init.normal_,
        data_dist = nn.init.normal_, 
        dimensions = conv2d_dimensions,
        hyperparameters = ConvParams(),
        device = device,
        deterministic = deterministic,
        autograd = False,
        dtype = None,
        iterations = iterations,
    )
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
