CONFIG = 'small_batch_600_steps_eval_logs'
BATCH_SIZE = 2

MAX_NUM_OF_WORDS = 64 # set limit for number of words to be considered from in a given sample (truncating after this point for now)
MAX_SLICE_DURATION = 96 # duration limit for per word frame

NUM_STEPS = 600
CHECKPOINT_INTERVAL = 200  # steps
LOG_INTERVAL = 50  # steps
EVALUATION_INTERVAL = 100  # steps
