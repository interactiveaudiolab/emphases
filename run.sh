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

# Experiments:
# - Transformer vs convolutional
#   - Layers
#   - Capacity
# - Features
#   - Loudness
#   - Pitch
#   - Periodicity
#   - Mels
# - BCE vs MSE loss
# - Resampling method
#   - max
#   - centerpoint
#   - mean
# - Architecture
#   - framewise
#   - posthoc-wordwise
#   - intermediate-wordwise

# Looped config generation and execution for first pass

python -m emphases.generate_configs

config_dir=config/hyperparam-search

# Loop through each file in the directory
for file in $(find "$config_dir" -type f -name "*.py" | sort -n); do
    # Get the file name without the directory path or extension
    file_name=$(basename "$file" .py)
    
    # Execute the command with the file name variable
    python -m emphases.train --config "$config_dir/$file_name.py" --gpus $1
done

# Second pass experiments

# Scaling laws
# TODO - configs (?)

# Ablations (features, resampling, and architecture)
# TODO - configs (22)

# Archived - First pass experiments - manual curation

# Start with a small, convolutional model (intermediate-wordwise + max + mels) and search loss ***
# python -m emphases.train --config config/first-pass/loss-bce-intermediate-wordwise-resampling-max-mels.py --gpus $1
# python -m emphases.train --config config/first-pass/loss-mse-intermediate-wordwise-resampling-max-mels.py --gpus $1

# Next, search features ***
# TODO - configs (15) - TODO - introduce USE_MELS flag
# configs (8) - if always using mels

# Mels
# Already covered in loss search

# # Mels + Loudness
# python -m emphases.train --config config/first-pass/feat-mels-loud-intermediate-wordwise-resampling-max.py --gpus $1

# # Mels + Pitch
# python -m emphases.train --config config/first-pass/feat-mels-pitch-intermediate-wordwise-resampling-max.py --gpus $1

# # Mels + Periodicity
# python -m emphases.train --config config/first-pass/feat-mels-period-intermediate-wordwise-resampling-max.py --gpus $1

# # Mels + Pitch + Loudness
# python -m emphases.train --config config/first-pass/feat-mels-pitch-loud-intermediate-wordwise-resampling-max.py --gpus $1

# # Mels + Periodicity + Loudness
# python -m emphases.train --config config/first-pass/feat-mels-period-loud-intermediate-wordwise-resampling-max.py --gpus $1

# # Mels + Pitch + Periodicity
# python -m emphases.train --config config/first-pass/feat-mels-pitch-period-intermediate-wordwise-resampling-max.py --gpus $1

# # Mels + Pitch + Periodicity + Loudness i.e. All
# python -m emphases.train --config config/first-pass/feat-all-intermediate-wordwise-resampling-max.py --gpus $1

# Next, search architecture and resampling method
# TODO - configs (7) using all features

# # Just Framewise
# python -m emphases.train --config config/first-pass/framewise-only.py --gpus $1

# # Max-Posthoc-wordwise
# python -m emphases.train --config config/first-pass/max-posthoc-wordwise.py --gpus $1

# # Centerpoint-Posthoc-wordwise
# python -m emphases.train --config config/first-pass/center-posthoc-wordwise.py --gpus $1

# # Avg-Posthoc-wordwise
# python -m emphases.train --config config/first-pass/avg-posthoc-wordwise.py --gpus $1

# # Max-Intermediate-wordwise
# python -m emphases.train --config config/first-pass/max-intermediate-wordwise.py --gpus $1

# # Centerpoint-Intermediate-wordwise
# python -m emphases.train --config config/first-pass/center-intermediate-wordwise.py --gpus $1

# # Avg-Intermediate-wordwise
# python -m emphases.train --config config/first-pass/avg-intermediate-wordwise.py --gpus $1

# Next, hparam search on both conv and transformer
# Done in the automated loop
# TODO - configs (?) - TODO loop operation (?)

# Evaluate baselines
python -m emphases.evaluate --config config/prominence.py --datasets buckeye
python -m emphases.evaluate --config config/pitch_variance.py --datasets buckeye
python -m emphases.evaluate --config config/duration_variance.py --datasets buckeye

# Plots
python -m emphases.plot --data plots/dataset_scaling.csv --x_label "Training data (seconds)" --output_file plots/dataset_scaling.jpg
python -m emphases.plot --data plots/num_annotators_scaling.csv --x_label "Number of annotators" --output_file plots/num_annotators_scaling.jpg


# Archived Configs
# Train and evaluate experiments
# python -m emphases.train --config config/framewise-linear-small.py --gpus $1
# python -m emphases.train --config config/framewise-nearest-small.py --gpus $1
# python -m emphases.train --config config/wordwise-small.py --gpus $1

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


