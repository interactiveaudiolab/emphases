
python -m emphases.train --config config/hparam-search/batch-025000.py --gpus $1
python -m emphases.train --config config/hparam-search/batch-075000.py --gpus $1
python -m emphases.train --config config/hparam-search/batch-100000.py --gpus $1
