# Runs experiments in the paper
# "Datasets and Scaling Laws for Neural Emphasis Prediction"

# Args
# $1 - index of GPU to use

# Download datasets
python -m emphases.data.download

# Setup experiments
python -m emphases.data.preprocess --gpu $1
python -m emphases.partition

# Train and evaluate experiments
python -m emphases.train --config config/framewise-linear-small.py --gpus $1
python -m emphases.train --config config/framewise-nearest-small.py --gpus $1
python -m emphases.train --config config/wordwise-small.py --gpus $1

# Experimental feature variations
# python -m emphases.train --config config/framewise-linear-mel-pitch-period-small.py --gpus $1
# python -m emphases.train --config config/framewise-linear-mel-loud-pitch-period-small.py --gpus $1
# python -m emphases.train --config config/framewise-linear-mel-prom-loud-pitch-period-small.py --gpus $1
# python -m emphases.train --config config/wordwise-mel-prom-loud-pitch-period-small.py --gpus $1
# python -m emphases.train --config config/framewise-linear-bceLoss-mel-prom-loud-pitch-period-small.py --gpus $1
# python -m emphases.train --config config/framewise-nearest-bceLoss-mel-prom-loud-pitch-period-small.py --gpus $1
# python -m emphases.train --config config/wordwise-bceLoss-mel-prom-loud-pitch-period-small.py --gpus $1

# Train on annotated dataset
# python -m emphases.train --config config/framewise-linear-annotated-mel-prom-loud-pitch-period-small.py --dataset annotate --gpus $1

# Evaluate baselines
python -m emphases.evaluate --config config/prominence.py --datasets buckeye
python -m emphases.evaluate --config config/pitch_variance.py --datasets buckeye
python -m emphases.evaluate --config config/duration_variance.py --datasets buckeye
