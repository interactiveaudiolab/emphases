import json

import torchaudio

import emphases


###############################################################################
# Loading utilities
###############################################################################


def audio(file):
    """Load audio and maybe resample"""
    # Load
    audio, sample_rate = torchaudio.load(file)

    # Maybe resample
    if sample_rate != emphases.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            sample_rate,
            emphases.SAMPLE_RATE)
        audio = resampler(audio)

    return audio

# NOTE - tgt is GPL licensed. We cannot use GPL code in our final codebase if
#        we want to release as MIT-licensed.
# def read_textgrid(filename, sample_rate=200):
#     import tgt
#     try:
#         tg = tgt.read_textgrid(filename) #, include_empty_intervals=True)
#     except:
#         print("reading "+filename+" failed")

#         return
#     tiers = []
#     labs = {}

#     for tier in tg.get_tier_names():
#         if (tg.get_tier_by_name(tier)).tier_type()!='IntervalTier':
#             continue
#         tiers.append(tg.get_tier_by_name(tier))

#         lab = []
#         for a in tiers[-1].annotations:

#             try:
#                 # this was for some past experiment
#                 if a.text in ["p1","p2","p3","p4","p5","p6","p7"]:
#                     lab[-1][-1]=lab[-1][-1]+"_"+a.text
#                 else:
#                 #lab.append([a.start_time*sample_rate,a.end_time*sample_rate,a.text.encode('utf-8')])
#                     lab.append([a.start_time*sample_rate,a.end_time*sample_rate,a.text])
#             except:
#                 pass
#             #print tiers[-1].encode('latin-1')
#         labs[tier.lower()] = lab
#     try:
#         for i in range(len(labs['prosody'])):
#             if labs['prosody'][i][2][-2:] not in ["p1","p2","p3","p4","p5","p6","p7"]:
#                 labs['prosody'][i][2]+="_p0"
#     except:
#         pass

#     return labs


def partition(dataset):
    """Load partitions for dataset"""
    with open(emphases.PARTITION_DIR / f'{dataset}.json') as file:
        return json.load(file)
