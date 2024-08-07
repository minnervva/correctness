All source code and inputs for all tests and programs reported in this paper can be found in our devoted github repository at
https://github.com/minnervva/correctness. The main directories `codes/test_reduce` and `codes/test-suite` contain the codes used in Section III and IV of the main text.

# Parallel sum

## Intallation

The c++ code can be found `codes/test_reduce` directory. The compilation only requires CMake, CUDA or
HIP. To compile it, clone the repository, go to the `codes/test_reduce/` directory and use the command

```bash
mkdir build && cd build
cmake -DREDUCE_SE_CUDA -DCMAKE_CUDA_ARCHITECTURES=70 ..
```
for support of the V100 or
```bash
mkdir build && cd build
cmake -DREDUCE_USE_HIP -DCMAKE_HIP_ARCHITECTURES=gfx90a ..
```
for Mi250X support. Set the variable `CMAKE_CUDA_ARCHITECTURES` or
`CMAKE_HIP_ARCHITECTURES` to the target GPU accordingly.

The `make` command will generate one executable named `test_reduce`. `test_reduce --help`
will return the full list of all available options.
## Usage
We run the following commands for the article
```bash
./test_reduce -S 1000000 --max_reduction_size 1000000 -A 10.0 -d uniform
./test_reduce -S 1000000 --max_reduction_size 1000000 -A 10.0 -d normal
./test_reduce -S 1000000 --max_reduction_size 1000000 -A 10.0 -d uniform -c 0.5
./test_reduce -S 500000 --max_reduction_size 1000000 -A 10.0 -d uniform --atomic_only
```
The executable generates csv files containing results of the variability
for each distribution. The name of the distribution and the GPU types
are included in the data file name.

The timings data are stored in a separate csv file, each line giving the
timings for different values of the parameters and name of the kernels.

The Mathematica directory contains mathematica files to explore the variability
data and generate the figures included in the github and the paper.

# PyTorch Tests

This directory holds all the PyTorch operation benchmark sweeps and the training and inference non-determinism tests.

## Repo organization

Please note the repository is organized as follows, with the operator benchmarks presented in the paper in `test-suite/paper-results` and the full sweep in the parent `test-suite`
```
test-suite/
| |-- data/
| |-- benchmark and plotting scripts
|-- paper-results/
  |-- data
  |-- benchmark and plotting scripts
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

Note, to add your own sets of hyperparameters to test for non-determinism, please head to the relevant scaffold.py file and edit the relevant operation hyperparameters and global variables in the main body.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
