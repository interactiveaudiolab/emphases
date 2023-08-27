# Runs experiments in the paper
# "Crowdsourced and Automatic Speech Prominence Estimation"

# Args
#  $1 - the GPU index


###########################
# Scaling law experiments #
###########################


## 1/64; 8 annotations
rm -rf data/cache/annotate/*
python -m emphases.data.download --datasets annotate --config config/scaling/64-8.py
python -m emphases.data.preprocess --datasets annotate --gpu $1
python -m emphases.partition --datasets annotate --config config/scaling/64-8.py
python -m emphases.train --config config/scaling/64-8.py --gpus $1 --dataset annotate

## 1/32; 4 annotations
rm -rf data/cache/annotate/*
python -m emphases.data.download --datasets annotate --config config/scaling/32-4.py
python -m emphases.data.preprocess --datasets annotate --gpu $1
python -m emphases.partition --datasets annotate --config config/scaling/32-4.py
python -m emphases.train --config config/scaling/32-4.py --gpus $1 --dataset annotate

## 1/16; 2 annotations
rm -rf data/cache/annotate/*
python -m emphases.data.download --datasets annotate --config config/scaling/16-2.py
python -m emphases.data.preprocess --datasets annotate --gpu $1
python -m emphases.partition --datasets annotate --config config/scaling/16-2.py
python -m emphases.train --config config/scaling/16-2.py --gpus $1 --dataset annotate

## 1/8; 1 annotations
rm -rf data/cache/annotate/*
python -m emphases.data.download --datasets annotate --config config/scaling/8-1.py
python -m emphases.data.preprocess --datasets annotate --gpu $1
python -m emphases.partition --datasets annotate --config config/scaling/8-1.py
python -m emphases.train --config config/scaling/8-1.py --gpus $1 --dataset annotate


#############
# Ablations #
#############


# Setup data
python -m emphases.data.download --datasets annotate
python -m emphases.data.preprocess --datasets annotate --gpu $1
python -m emphases.partition --datasets annotate

# Start with a small, transformer model (intermediate-wordwise + max + mels + prosody) and search loss
python -m emphases.train --config config/base.py --gpus $1 --dataset annotate
python -m emphases.train --config config/bce.py --gpus $1 --dataset annotate

# Next, search features
python -m emphases.train --config config/ablate-features/no-loudness.py --gpus $1 --dataset annotate
python -m emphases.train --config config/ablate-features/no-mels.py --gpus $1 --dataset annotate
python -m emphases.train --config config/ablate-features/no-pitch.py --gpus $1 --dataset annotate

# Next, search combinations of downsampling method and location
python -m emphases.train --config config/ablate-resampling/average-inference.py --gpus $1 --dataset annotate
python -m emphases.train --config config/ablate-resampling/average-intermediate.py --gpus $1 --dataset annotate
python -m emphases.train --config config/ablate-resampling/average-loss.py --gpus $1 --dataset annotate
python -m emphases.train --config config/ablate-resampling/center-inference.py --gpus $1 --dataset annotate
python -m emphases.train --config config/ablate-resampling/center-intermediate.py --gpus $1 --dataset annotate
python -m emphases.train --config config/ablate-resampling/center-loss.py --gpus $1 --dataset annotate
python -m emphases.train --config config/ablate-resampling/max-inference.py --gpus $1 --dataset annotate
python -m emphases.train --config config/ablate-resampling/max-loss.py --gpus $1 --dataset annotate

# Next, hparam search on both conv and transformer
python -m emphases.train --config config/hparam-search/convolution-4-128.py --gpus $1 --dataset annotate
python -m emphases.train --config config/hparam-search/convolution-4-256.py --gpus $1 --dataset annotate
python -m emphases.train --config config/hparam-search/convolution-4-512.py --gpus $1 --dataset annotate
python -m emphases.train --config config/hparam-search/convolution-6-128.py --gpus $1 --dataset annotate
python -m emphases.train --config config/hparam-search/convolution-6-256.py --gpus $1 --dataset annotate
python -m emphases.train --config config/hparam-search/convolution-6-512.py --gpus $1 --dataset annotate
python -m emphases.train --config config/hparam-search/transformer-2-256.py --gpus $1 --dataset annotate
python -m emphases.train --config config/hparam-search/transformer-2-512.py --gpus $1 --dataset annotate

# Ablations (features, resampling, and architecture)
# TODO - configs (22)


#############
# Baselines #
#############


# Evaluate baselines
python -m emphases.evaluate --config config/prominence.py
python -m emphases.evaluate --config config/pitch-variance.py
python -m emphases.evaluate --config config/duration-variance.py


############
# Analysis #
############


# Analyze the data
# python -m emphases.data.analyze --dataset annotate

# Plots
# python -m emphases.plot.scaling \
#     --runs 8-1 16-2 32-4 64-8 \
#     --x_label "Annotators per utterance" \
#     --output_file plots/num_annotators_scaling.jpg
