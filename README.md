# Quantum Random Walk Simulator

## Setup
```
conda env create -f environment.yml
```
will create a QRW conda environment containing cirq and numpy.

## Usage
Execute 
```
conda activate QRW
python quantum_random_walk.py
```
to simulate a quantum random walk with default parameters. 

A variety of command line arguments are supported to customize the random walk. To view use
`python quantum_random_walk.py -h ` 

Of particular note is the `--classical` argument. If supplied, a measurement is made at each step of the walk, destroying the interference effects that characterize the QRW. Recommend using small `--n_walks` (<100) when the `--classical` argument is passed.

