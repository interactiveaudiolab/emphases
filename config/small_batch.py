CONFIG = 'small_batch'
BATCH_SIZE = 2

MAX_NUM_OF_WORDS = 64 # set limit for number of words to be considered from in a given sample (truncating after this point for now)
MAX_WORD_DURATION = 96 # duration limit for per word frame

NUM_STEPS = 16
CHECKPOINT_INTERVAL = 4  # steps
LOG_INTERVAL = 4  # steps
EVALUATION_INTERVAL = 4  # steps
