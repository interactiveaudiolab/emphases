python -m emphases.train --config config/downsample/sum-inference.py --gpus $1
python -m emphases.train --config config/downsample/sum-intermediate.py --gpus $1
python -m emphases.train --config config/downsample/sum-input.py --gpus $1
python -m emphases.train --config config/downsample/sum-loss.py --gpus $1
