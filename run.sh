# Runs experiments in the paper
# "Datasets and Scaling Laws for Automatic Emphasis Prediction"

# Args
#  $1 - the GPU index

# # Download datasets
# python -m emphases.data.download

# # Setup experiments
# python -m emphases.data.preprocess --gpu $1
# python -m emphases.partition

# Start with a small, transformer model (intermediate-wordwise + max + mels + prosody) and search loss
python -m emphases.train --config config/first-pass/base.py --gpus $1
python -m emphases.train --config config/first-pass/mse.py --gpus $1

# Next, search features
python -m emphases.train --config config/first-pass/no-loudness.py --gpus $1
python -m emphases.train --config config/first-pass/no-mels.py --gpus $1
python -m emphases.train --config config/first-pass/no-periodicity.py --gpus $1
python -m emphases.train --config config/first-pass/no-pitch.py --gpus $1

# Next, search combinations of downsampling method and location
python -m emphases.train --config config/first-pass/average-inference.py --gpus $1
python -m emphases.train --config config/first-pass/average-intermediate.py --gpus $1
python -m emphases.train --config config/first-pass/average-loss.py --gpus $1
python -m emphases.train --config config/first-pass/center-inference.py --gpus $1
python -m emphases.train --config config/first-pass/center-intermediate.py --gpus $1
python -m emphases.train --config config/first-pass/center-loss.py --gpus $1
python -m emphases.train --config config/first-pass/max-inference.py --gpus $1
python -m emphases.train --config config/first-pass/max-loss.py --gpus $1

# Next, hparam search on both conv and transformer
# python -m emphases.train --config config/first-pass/convolution-4-256.py --gpus $1
# python -m emphases.train --config config/first-pass/convolution-4-512.py --gpus $1
# python -m emphases.train --config config/first-pass/convolution-6-1024.py --gpus $1
# python -m emphases.train --config config/first-pass/convolution-6-256.py --gpus $1
# python -m emphases.train --config config/first-pass/convolution-6-512.py --gpus $1
# python -m emphases.train --config config/first-pass/convolution-6-1024.py --gpus $1
# python -m emphases.train --config config/first-pass/transformer-4-256.py --gpus $1
# python -m emphases.train --config config/first-pass/transformer-4-512.py --gpus $1
# python -m emphases.train --config config/first-pass/transformer-6-256.py --gpus $1
# python -m emphases.train --config config/first-pass/transformer-6-512.py --gpus $1


# Second pass experiments

# Scaling laws
# TODO - configs (?)

# Ablations (features, resampling, and architecture)
# TODO - configs (22)

# Evaluate baselines
# TODO - use all datasets during final run
# python -m emphases.evaluate --config config/prominence.py --datasets buckeye
# python -m emphases.evaluate --config config/pitch-variance.py --datasets buckeye
# python -m emphases.evaluate --config config/duration-variance.py --datasets buckeye

# Analyze the annotated data
# python -m emphases.data.analyze --dataset annotate

# Plots
# python -m emphases.plot.scaling \
#     --data plots/dataset_scaling.csv \
#     --x_label "Training data (seconds)" \
#     --output_file plots/dataset_scaling.jpg
# python -m emphases.plot.scaling \
#     --data plots/num_annotators_scaling.csv \
#     --x_label "Number of annotators" \
#     --output_file plots/num_annotators_scaling.jpg
