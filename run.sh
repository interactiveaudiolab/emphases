# Runs experiments in the paper
# "Datasets and Scaling Laws for Neural Emphasis Prediction"

# Args
#  $1 - index of GPU to use

# Download datasets
python -m emphases.data.download

# Setup experiments
python -m emphases.data.preprocess --gpu $1
python -m emphases.partition

# Analyze the annotated data
python -m emphases.data.analyze --dataset annotate

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

# Looped config generation and execution for first pass
# python -m emphases.generate_configs

# config_dir=config/hyperparam-search

# # Loop through each file in the directory
# for file in $(find "$config_dir" -type f -name "*.py" | sort -n); do
#     # Get the file name without the directory path or extension
#     file_name=$(basename "$file" .py)

#     # Execute the command with the file name variable
#     python -m emphases.train --config "$config_dir/$file_name.py" --gpus $1
# done

# Second pass experiments

# Scaling laws
# TODO - configs (?)

# Ablations (features, resampling, and architecture)
# TODO - configs (22)

# Evaluate baselines
python -m emphases.evaluate --config config/prominence.py --datasets buckeye
python -m emphases.evaluate --config config/pitch_variance.py --datasets buckeye
python -m emphases.evaluate --config config/duration_variance.py --datasets buckeye

# Plots
python -m emphases.plot \
    --data plots/dataset_scaling.csv \
    --x_label "Training data (seconds)" \
    --output_file plots/dataset_scaling.jpg
python -m emphases.plot \
    --data plots/num_annotators_scaling.csv \
    --x_label "Number of annotators" \
    --output_file plots/num_annotators_scaling.jpg
