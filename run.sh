# Runs experiments in the paper
# "Datasets and Scaling Laws for Automatic Emphasis Prediction"

# Args
#  $1 - the GPU index

# Download datasets
python -m emphases.data.download --datasets annotate

# Analyze the data
# python -m emphases.data.analyze --dataset annotate

# Setup experiments
python -m emphases.data.preprocess --datasets annotate --gpu $1
python -m emphases.partition --datasets annotate

# Start with a small, transformer model (intermediate-wordwise + max + mels + prosody) and search loss
python -m emphases.train --config config/base.py --gpus $1 --dataset annotate
# python -m emphases.train --config config/bce.py --gpus $1 --dataset annotate

# Next, search features
# python -m emphases.train --config config/ablate-features/no-loudness.py --gpus $1 --dataset annotate
# python -m emphases.train --config config/ablate-features/no-mels.py --gpus $1 --dataset annotate
# python -m emphases.train --config config/ablate-features/no-periodicity.py --gpus $1 --dataset annotate
# python -m emphases.train --config config/ablate-features/no-pitch.py --gpus $1 --dataset annotate

# Next, search combinations of downsampling method and location
# python -m emphases.train --config config/ablate-resmpling/average-inference.py --gpus $1 --dataset annotate
# python -m emphases.train --config config/ablate-resmpling/average-intermediate.py --gpus $1 --dataset annotate
# python -m emphases.train --config config/ablate-resmpling/average-loss.py --gpus $1 --dataset annotate
# python -m emphases.train --config config/ablate-resmpling/center-inference.py --gpus $1 --dataset annotate
# python -m emphases.train --config config/ablate-resmpling/center-intermediate.py --gpus $1 --dataset annotate
# python -m emphases.train --config config/ablate-resmpling/center-loss.py --gpus $1 --dataset annotate
# python -m emphases.train --config config/ablate-resmpling/max-inference.py --gpus $1 --dataset annotate
# python -m emphases.train --config config/ablate-resmpling/max-loss.py --gpus $1 --dataset annotate

# Next, hparam search on both conv and transformer
# python -m emphases.train --config config/hparam-search/convolution-4-256.py --gpus $1 --dataset annotate
# python -m emphases.train --config config/hparam-search/convolution-4-512.py --gpus $1 --dataset annotate
# python -m emphases.train --config config/hparam-search/convolution-4-1024.py --gpus $1 --dataset annotate
# python -m emphases.train --config config/hparam-search/convolution-6-256.py --gpus $1 --dataset annotate
# python -m emphases.train --config config/hparam-search/convolution-6-512.py --gpus $1 --dataset annotate
# python -m emphases.train --config config/hparam-search/convolution-6-1024.py --gpus $1 --dataset annotate
# python -m emphases.train --config config/hparam-search/transformer-4-256.py --gpus $1 --dataset annotate
# python -m emphases.train --config config/hparam-search/transformer-4-512.py --gpus $1 --dataset annotate
# python -m emphases.train --config config/hparam-search/transformer-6-256.py --gpus $1 --dataset annotate
# python -m emphases.train --config config/hparam-search/transformer-6-512.py --gpus $1 --dataset annotate

# Ablations (features, resampling, and architecture)
# TODO - configs (22)

# Evaluate baselines
# python -m emphases.evaluate --config config/prominence.py
# python -m emphases.evaluate --config config/pitch-variance.py
# python -m emphases.evaluate --config config/duration-variance.py

# Scaling laws

# ## 1/64; 8 annotations
# rm -rf data/cache/annotate/*
# python -m emphases.data.download --datasets annotate --config config/scaling/64-8.py
# python -m emphases.data.preprocess --datasets annotate --gpu $1
# python -m emphases.partition --datasets annotate --config config/scaling/64-8.py
# python -m emphases.train --config config/scaling/64-8.py --gpus $1 --dataset annotate

# ## 1/32; 4 annotations
# rm -rf data/cache/annotate/*
# python -m emphases.data.download --datasets annotate --config config/scaling/32-4.py
# python -m emphases.data.preprocess --datasets annotate --gpu $1
# python -m emphases.partition --datasets annotate --config config/scaling/32-4.py
# python -m emphases.train --config config/scaling/32-4.py --gpus $1 --dataset annotate

# ## 1/16; 2 annotations
# rm -rf data/cache/annotate/*
# python -m emphases.data.download --datasets annotate --config config/scaling/16-2.py
# python -m emphases.data.preprocess --datasets annotate --gpu $1
# python -m emphases.partition --datasets annotate --config config/scaling/16-2.py
# python -m emphases.train --config config/scaling/16-2.py --gpus $1 --dataset annotate

# ## 1/8; 1 annotations
# rm -rf data/cache/annotate/*
# python -m emphases.data.download --datasets annotate --config config/scaling/8-1.py
# python -m emphases.data.preprocess --datasets annotate --gpu $1
# python -m emphases.partition --datasets annotate --config config/scaling/8-1.py
# python -m emphases.train --config config/scaling/8-1.py --gpus $1 --dataset annotate

# Plots
# python -m emphases.plot.scaling \
#     --runs 8-1 16-2 32-4 64-8 \
#     --x_label "Annotators per utterance" \
#     --output_file plots/num_annotators_scaling.jpg
