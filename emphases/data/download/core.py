import os
import emphases
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import requests, zipfile, io
from tqdm import tqdm

def datasets(datasets):
    # assuming only one dataset is passed
    DATA_DIR = emphases.DATA_DIR / datasets[0]
    # DATA_DIR = os.path.join(DATA_DIR, "Buckeye")

    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)
    
    # BuckEye corpus - https://buckeyecorpus.osu.edu/php/speech.php?PHPSESSID=njvutg9l90fc3ebg7v30vnhmk1
    # speakers under consideration
    speakers = {
        's02-1': ["https://buckeyecorpus.osu.edu/speechfiles/s02/s0201a.zip", "https://buckeyecorpus.osu.edu/speechfiles/s02/s0201b.zip"],
        's03-1': ["https://buckeyecorpus.osu.edu/speechfiles/s03/s0301a.zip", "https://buckeyecorpus.osu.edu/speechfiles/s03/s0301b.zip"],
        's04-1': ["https://buckeyecorpus.osu.edu/speechfiles/s04/s0401a.zip", "https://buckeyecorpus.osu.edu/speechfiles/s04/s0401b.zip"],
        's10-1': ["https://buckeyecorpus.osu.edu/speechfiles/s10/s1001a.zip", "https://buckeyecorpus.osu.edu/speechfiles/s10/s1001b.zip"],
        's11-1': ["https://buckeyecorpus.osu.edu/speechfiles/s11/s1101a.zip", "https://buckeyecorpus.osu.edu/speechfiles/s11/s1101b.zip"],
        's14-1': ["https://buckeyecorpus.osu.edu/speechfiles/s14/s1401a.zip", "https://buckeyecorpus.osu.edu/speechfiles/s14/s1401b.zip"],
        's16-1': ["https://buckeyecorpus.osu.edu/speechfiles/s16/s1601a.zip", "https://buckeyecorpus.osu.edu/speechfiles/s16/s1601b.zip"],
        's17-1': ["https://buckeyecorpus.osu.edu/speechfiles/s17/s1701a.zip", "https://buckeyecorpus.osu.edu/speechfiles/s17/s1701b.zip"],
        's21-1': ["https://buckeyecorpus.osu.edu/speechfiles/s21/s2101a.zip", "https://buckeyecorpus.osu.edu/speechfiles/s21/s2101b.zip"],
        's22-1': ["https://buckeyecorpus.osu.edu/speechfiles/s22/s2201a.zip", "https://buckeyecorpus.osu.edu/speechfiles/s22/s2201b.zip"],
        's24-1': ["https://buckeyecorpus.osu.edu/speechfiles/s24/s2401a.zip", "https://buckeyecorpus.osu.edu/speechfiles/s24/s2401b.zip"],
        's25-1': ["https://buckeyecorpus.osu.edu/speechfiles/s25/s2501a.zip", "https://buckeyecorpus.osu.edu/speechfiles/s25/s2501b.zip"],
        's26-1': ["https://buckeyecorpus.osu.edu/speechfiles/s26/s2601a.zip", "https://buckeyecorpus.osu.edu/speechfiles/s26/s2601b.zip"],
        's32-1': ["https://buckeyecorpus.osu.edu/speechfiles/s32/s3201a.zip", "https://buckeyecorpus.osu.edu/speechfiles/s32/s3201b.zip"],
        's33-1': ["https://buckeyecorpus.osu.edu/speechfiles/s33/s3301a.zip", "https://buckeyecorpus.osu.edu/speechfiles/s33/s3301b.zip"],
        's35-1': ["https://buckeyecorpus.osu.edu/speechfiles/s35/s3501a.zip", "https://buckeyecorpus.osu.edu/speechfiles/s35/s3501b.zip"]
    }

    for speaker_id in tqdm(speakers):
        SPEAKER_FOLDER = os.path.join(DATA_DIR, speaker_id)
        if not os.path.isdir(SPEAKER_FOLDER):
            os.mkdir(SPEAKER_FOLDER)
            for link in speakers[speaker_id]:
                r = requests.get(link)
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(SPEAKER_FOLDER)
