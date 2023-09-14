# Runs experiments in the paper
# "Crowdsourced and Automatic Speech Prominence Estimation"

# Args
#  $1 - the GPU index


####################################
# Annotator redundancy experiments #
####################################


# 1/64; 8 annotations
rm -rf data/cache/crowdsource/*
python -m emphases.data.download --config config/scaling/64-8.py
python -m emphases.data.preprocess --gpu $1
python -m emphases.partition --config config/scaling/64-8.py
python -m emphases.train --config config/scaling/64-8.py --gpus $1

# 1/32; 4 annotations
rm -rf data/cache/crowdsource/*
python -m emphases.data.download --config config/scaling/32-4.py
python -m emphases.data.preprocess --gpu $1
python -m emphases.partition --config config/scaling/32-4.py
python -m emphases.train --config config/scaling/32-4.py --gpus $1

# 1/16; 2 annotations
rm -rf data/cache/crowdsource/*
python -m emphases.data.download --config config/scaling/16-2.py
python -m emphases.data.preprocess --gpu $1
python -m emphases.partition --config config/scaling/16-2.py
python -m emphases.train --config config/scaling/16-2.py --gpus $1

# 1/8; 1 annotations
rm -rf data/cache/crowdsource/*
python -m emphases.data.download --config config/scaling/8-1.py
python -m emphases.data.preprocess --gpu $1
python -m emphases.partition --config config/scaling/8-1.py
python -m emphases.train --config config/scaling/8-1.py --gpus $1

# Plot results
python -m emphases.plot.scaling \
    --evaluations 8-1 16-2 32-4 64-8 \
    --xlabel "Annotators per utterance" \
    --output_file results/scaling-annotators.pdf \
    --sizes 3200 1600 800 400 \
    --scores 0.686 0.683 0.667 0.664 \
    --steps 967 933 567 467 \
    --yticks 0.66 0.67 0.68 0.69 \
    --text_offsets 0.007 0.01 0.007 0.007


####################################
# Dataset size scaling experiments #
####################################


# 400 utterances
rm -rf data/cache/crowdsource/*
python -m emphases.data.download --config config/scaling/400.py
python -m emphases.data.preprocess --gpu $1
python -m emphases.partition --config config/scaling/400.py
python -m emphases.train --config config/scaling/400.py --gpus $1

# 800 utterances
rm -rf data/cache/crowdsource/*
python -m emphases.data.download --config config/scaling/800.py
python -m emphases.data.preprocess --gpu $1
python -m emphases.partition --config config/scaling/800.py
python -m emphases.train --config config/scaling/800.py --gpus $1

# 1600 utterances
rm -rf data/cache/crowdsource/*
python -m emphases.data.download --config config/scaling/1600.py
python -m emphases.data.preprocess --gpu $1
python -m emphases.partition --config config/scaling/1600.py
python -m emphases.train --config config/scaling/1600.py --gpus $1

# 3200 utterances
rm -rf data/cache/crowdsource/*
python -m emphases.data.download --config config/scaling/3200.py
python -m emphases.data.preprocess --gpu $1
python -m emphases.partition --config config/scaling/3200.py
python -m emphases.train --config config/scaling/3200.py --gpus $1

# Plot results
python -m emphases.plot.scaling \
    --evaluations 400 800 1600 3200 \
    --xlabel Utterances \
    --output_file results/scaling-data.pdf \
    --yticks 0.63 0.65 0.67 0.69 \
    --scores 0.633 0.657 0.678 0.687 \
    --steps 400 500 767 1433 \
    --text_offsets 0.007 0.007 0.007 0.007


##############
# Best model #
##############


python -m emphases.data.download
python -m emphases.data.preprocess --gpu $1
python -m emphases.partition
python -m emphases.train --config config/base.py --gpus $1


#############
# Ablations #
#############


python -m emphases.train --config config/mse.py --gpus $1


##############
# Downsample #
##############


python -m emphases.train --config config/downsample/average-inference.py --gpus $1
python -m emphases.train --config config/downsample/average-intermediate.py --gpus $1
python -m emphases.train --config config/downsample/average-input.py --gpus $1
python -m emphases.train --config config/downsample/average-loss.py --gpus $1
python -m emphases.train --config config/downsample/center-inference.py --gpus $1
python -m emphases.train --config config/downsample/center-intermediate.py --gpus $1
python -m emphases.train --config config/downsample/center-input.py --gpus $1
python -m emphases.train --config config/downsample/center-loss.py --gpus $1
python -m emphases.train --config config/downsample/max-inference.py --gpus $1
python -m emphases.train --config config/downsample/max-intermediate.py --gpus $1
python -m emphases.train --config config/downsample/max-input.py --gpus $1
python -m emphases.train --config config/downsample/max-loss.py --gpus $1


####################################
# Large-scale automatic annotation #
####################################


python -m emphases.data.download --datasets automatic --gpu $1
python -m emphases.partition --datasets automatic
python -m emphases.data.preprocess --datasets automatic --gpu $1
python -m emphases.train --config config/scaling/base-automatic-128.py --dataset automatic --gpus $1


#############
# Baselines #
#############


python -m emphases.evaluate --config config/baselines/prominence.py
