# Runs experiments in the paper
# "Datasets and Scaling Laws for Neural Emphasis Prediction"

# Args
# $1 - list of indices of GPUs to use

# Download datasets
python -m emphases.data.download

# Setup experiments
python -m emphases.data.preprocess
python -m emphases.partition

# Run experiments
python -m emphases.train --config config/framewise-linear-small.py --gpus $1
python -m emphases.train --config config/framewise-nearest-small.py --gpus $1
python -m emphases.train --config config/wordwise-small.py --gpus $1
