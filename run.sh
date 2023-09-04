# Runs experiments in the paper
# "Crowdsourced and Automatic Speech Prominence Estimation"

# Args
#  $1 - the GPU index


####################################
# Annotator redundancy experiments #
####################################


# ## 1/64; 8 annotations
# rm -rf data/cache/annotate/*
# python -m emphases.data.download --datasets annotate --config config/scaling/64-8.py
# python -m emphases.data.preprocess --datasets annotate --gpu $1
# python -m emphases.partition --datasets annotate --config config/scaling/64-8.py
# python -m emphases.train --config config/scaling/64-8.py --gpus $1

# ## 1/32; 4 annotations
# rm -rf data/cache/annotate/*
# python -m emphases.data.download --datasets annotate --config config/scaling/32-4.py
# python -m emphases.data.preprocess --datasets annotate --gpu $1
# python -m emphases.partition --datasets annotate --config config/scaling/32-4.py
# python -m emphases.train --config config/scaling/32-4.py --gpus $1

# ## 1/16; 2 annotations
# rm -rf data/cache/annotate/*
# python -m emphases.data.download --datasets annotate --config config/scaling/16-2.py
# python -m emphases.data.preprocess --datasets annotate --gpu $1
# python -m emphases.partition --datasets annotate --config config/scaling/16-2.py
# python -m emphases.train --config config/scaling/16-2.py --gpus $1

# ## 1/8; 1 annotations
# rm -rf data/cache/annotate/*
# python -m emphases.data.download --datasets annotate --config config/scaling/8-1.py
# python -m emphases.data.preprocess --datasets annotate --gpu $1
# python -m emphases.partition --datasets annotate --config config/scaling/8-1.py
# python -m emphases.train --config config/scaling/8-1.py --gpus $1

# ## > 1 annotations
# rm -rf data/cache/annotate/*
# python -m emphases.data.download --datasets annotate --config config/scaling/min-2.py
# python -m emphases.data.preprocess --datasets annotate --gpu $1
# python -m emphases.partition --datasets annotate --config config/scaling/min-2.py
# python -m emphases.train --config config/scaling/min-2.py --gpus $1

# # Plot results
# python -m emphases.plot.scaling \
#     --evaluations 8-1 16-2 32-4 64-8 \
#     --xlabel "Annotators per utterance" \
#     --output_file results/scaling-annotators.pdf \
#     --sizes 3200 1600 800 400 \
#     --yticks 0.67 0.68 0.69


####################################
# Dataset size scaling experiments #
####################################


# ## 400 utterances
# rm -rf data/cache/annotate/*
# python -m emphases.data.download --datasets annotate --config config/scaling/400.py
# python -m emphases.data.preprocess --datasets annotate --gpu $1
# python -m emphases.partition --datasets annotate --config config/scaling/400.py
# python -m emphases.train --config config/scaling/400.py --gpus $1

# ## 800 utterances
# rm -rf data/cache/annotate/*
# python -m emphases.data.download --datasets annotate --config config/scaling/800.py
# python -m emphases.data.preprocess --datasets annotate --gpu $1
# python -m emphases.partition --datasets annotate --config config/scaling/800.py
# python -m emphases.train --config config/scaling/800.py --gpus $1

# ## 1600 utterances
# rm -rf data/cache/annotate/*
# python -m emphases.data.download --datasets annotate --config config/scaling/1600.py
# python -m emphases.data.preprocess --datasets annotate --gpu $1
# python -m emphases.partition --datasets annotate --config config/scaling/1600.py
# python -m emphases.train --config config/scaling/1600.py --gpus $1

# ## 3200 utterances
# rm -rf data/cache/annotate/*
# python -m emphases.data.download --datasets annotate --config config/scaling/3200.py
# python -m emphases.data.preprocess --datasets annotate --gpu $1
# python -m emphases.partition --datasets annotate --config config/scaling/3200.py
# python -m emphases.train --config config/scaling/3200.py --gpus $1

# # Plot results
# python -m emphases.plot.scaling \
#     --evaluations 400 800 1600 3200 \
#     --xlabel Utterances \
#     --output_file results/scaling-data.pdf \
#     --yticks 0.55 0.60 0.65 0.70 0.75


##########################
# Normalization ablation #
##########################


rm -rf data/cache/annotate/*
python -m emphases.data.download --datasets annotate --config config/ablate-features/normalize.py
python -m emphases.data.preprocess --datasets annotate --gpu $1 --config config/ablate-features/normalize.py
python -m emphases.partition --datasets annotate --config config/ablate-features/normalize.py
python -m emphases.train --config config/ablate-features/normalize.py --gpus $1


#############
# Ablations #
#############


# Setup data
python -m emphases.data.download --datasets annotate
python -m emphases.data.preprocess --datasets annotate --gpu $1
python -m emphases.partition --datasets annotate

# TEMPORARY
python -m emphases.train --config config/hparam-search/buckets-1.py --gpus $1
python -m emphases.train --config config/hparam-search/buckets-2.py --gpus $1
python -m emphases.train --config config/hparam-search/dropout-05.py --gpus $1
python -m emphases.train --config config/hparam-search/dropout-10.py --gpus $1

# # Start with a small, transformer model (intermediate-wordwise + max + mels + prosody) and search loss
python -m emphases.train --config config/base.py --gpus $1
python -m emphases.train --config config/bce.py --gpus $1

# # Next, search features
# python -m emphases.train --config config/ablate-features/no-loudness.py --gpus $1
# python -m emphases.train --config config/ablate-features/no-mels.py --gpus $1

# # Next, search combinations of downsampling method and location
# python -m emphases.train --config config/ablate-resampling/average-inference.py --gpus $1
# python -m emphases.train --config config/ablate-resampling/average-intermediate.py --gpus $1
# python -m emphases.train --config config/ablate-resampling/average-input.py --gpus $1
# python -m emphases.train --config config/ablate-resampling/average-loss.py --gpus $1
# python -m emphases.train --config config/ablate-resampling/center-inference.py --gpus $1
# python -m emphases.train --config config/ablate-resampling/center-intermediate.py --gpus $1
# python -m emphases.train --config config/ablate-resampling/center-input.py --gpus $1
# python -m emphases.train --config config/ablate-resampling/center-loss.py --gpus $1
# python -m emphases.train --config config/ablate-resampling/max-inference.py --gpus $1
# python -m emphases.train --config config/ablate-resampling/max-input.py --gpus $1
# python -m emphases.train --config config/ablate-resampling/max-loss.py --gpus $1
# python -m emphases.train --config config/ablate-resampling/sum-inference.py --gpus $1
# python -m emphases.train --config config/ablate-resampling/sum-intermediate.py --gpus $1
# python -m emphases.train --config config/ablate-resampling/sum-input.py --gpus $1
# python -m emphases.train --config config/ablate-resampling/sum-loss.py --gpus $1

# # Next, hparam search on both conv and transformer
# python -m emphases.train --config config/hparam-search/convolution-4-64.py --gpus $1
# python -m emphases.train --config config/hparam-search/convolution-4-128.py --gpus $1
# python -m emphases.train --config config/hparam-search/convolution-4-256.py --gpus $1
# python -m emphases.train --config config/hparam-search/convolution-6-64.py --gpus $1
# python -m emphases.train --config config/hparam-search/convolution-6-128.py --gpus $1
# python -m emphases.train --config config/hparam-search/convolution-6-256.py --gpus $1
# python -m emphases.train --config config/hparam-search/transformer-2-64.py --gpus $1
# python -m emphases.train --config config/hparam-search/transformer-2-128.py --gpus $1
# python -m emphases.train --config config/hparam-search/transformer-2-256.py --gpus $1


#############
# Baselines #
#############


# Evaluate baselines
# python -m emphases.evaluate --config config/prominence.py
# python -m emphases.evaluate --config config/pitch-variance.py --gpu $1
# python -m emphases.evaluate --config config/duration-variance.py


############
# Analysis #
############


# Analyze the data
# python -m emphases.data.analyze
