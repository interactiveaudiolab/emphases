import itertools
from pathlib import Path

def generate_configs():
    SAVE_DIR = Path("config/hyperparam-search")
    SAVE_DIR.mkdir(exist_ok=True)

    # Define all possible values for each parameter
    # TODO: Layers, Capacity
    ENCODING_STACK = ['conv', 'transformer']
    features = {
        'loudness': [True, False],
        'pitch': [True, False],
        'periodicity': [True, False],
        # 'mels': [True, False], # TODO
        'mels': [True]
    }
    LOSS_FUNCTION = ['BCELogitloss', 'MSEloss']
    FRAMES_TO_WORDS_RESAMPLE = ['max', 'center', 'avg']
    ARCHITECTURE = ['framewise', 'posthoc-wordwise', 'intermediate-wordwise']

    # Define default values for non-variable parameters
    MODULE = 'emphases'
    DATASETS = ['annotate']
    BATCH_SIZE = 2
    NUM_STEPS = 1000
    CHECKPOINT_INTERVAL = 200
    LOG_INTERVAL = 50
    EVALUATION_INTERVAL = 50

    # Generate all possible combinations of values
    combinations = list(itertools.product(
                        ENCODING_STACK,
                        features['loudness'],
                        features['pitch'],
                        features['periodicity'],
                        features['mels'],
                        LOSS_FUNCTION, 
                        FRAMES_TO_WORDS_RESAMPLE, 
                        ARCHITECTURE))

    # Loop through all combinations and generate a config file for each
    for idx, comb in enumerate(combinations):
        # Unpack the combination values
        encoding_stack, loudness_feature, pitch_feature, periodicity_feature, mels_feature, loss_function, resample_method, architecture = comb

        # Create a unique prefix based on parameter values
        prefix = f"{architecture}_{encoding_stack}"
        if loudness_feature: prefix += '_loudness'
        if pitch_feature: prefix += '_pitch'
        if periodicity_feature: prefix += '_periodicity'
        if mels_feature: prefix += '_mels'
        prefix += f"_{loss_function}_{resample_method}"

        config = {}
        config['MODULE'] = MODULE
        config['CONFIG'] = f'{prefix}'
        config['METHOD'] = 'wordwise' if architecture == 'intermediate-wordwise' else 'framewise'
        config['ENCODING_STACK'] = encoding_stack
        config['MODEL_TO_WORDS'] = True if architecture != 'framewise' else False
        config['FRAMES_TO_WORDS_RESAMPLE'] = resample_method
        config['PITCH_FEATURE'] = pitch_feature
        config['PERIODICITY_FEATURE'] = periodicity_feature
        config['LOUDNESS_FEATURE'] = loudness_feature

        # Training specific params
        config['BATCH_SIZE'] = BATCH_SIZE
        config['USE_BCE_LOGITS_LOSS'] = True if loss_function == 'BCE' else False
        config['DATASETS'] = DATASETS
        config['NUM_STEPS'] = NUM_STEPS
        config['CHECKPOINT_INTERVAL'] = CHECKPOINT_INTERVAL
        config['LOG_INTERVAL'] = LOG_INTERVAL
        config['EVALUATION_INTERVAL'] = EVALUATION_INTERVAL

        # Save the config file
        filename = SAVE_DIR / f'config_{idx}.py'
        with open(filename, 'w') as f:
            for key, value in config.items():
                if isinstance(value, str):
                    f.write(f"{key} = '{value}'\n")
                else:
                    f.write(f"{key} = {value}\n")

if __name__=="__main__":
    generate_configs()
