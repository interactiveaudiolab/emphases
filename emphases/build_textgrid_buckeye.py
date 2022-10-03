import os
from praatio import textgrid

def build_textgrid(word_file, phones_file):
    
    basename = word_file.split('/')[-1].replace('.words', '')
    
    with open(word_file) as f:
        data = f.read()
        f.close()
        
    with open(phones_file) as f:
        data_phones = f.read()
        f.close()
        
    raw_data = [x.strip().split(';')[0].split() for x in data.split('\n')[9:-1]]
    
    raw_data_phones = [x.strip().split(';')[0].split() for x in data_phones.split('\n')[9:-1]]
    
    grid_word_tuples = []
    if raw_data[0][-1]=='{B_TRANS}' and raw_data[-1][-1]=='{E_TRANS}':
        start = 0.0
        for row in raw_data:
            end = row[0]
            tup = (float(start), float(end), row[-1])
            grid_word_tuples.append(tup)
            start = row[0]
    else:
        print("invalid .words file, aborting formation, grid will be empty")

    grid_phones_tuples = []
    if raw_data_phones[0][-1]=='{B_TRANS}' and raw_data_phones[-1][-1]=='{E_TRANS}':
        start = 0.0
        for row in raw_data_phones:
            end = row[0]
            tup = (float(start), float(end), row[-1])
            grid_phones_tuples.append(tup)
            start = row[0]
    else:
        print("invalid .phones file, aborting formation, grid will be empty")
    
    # Build the grids
    tg = textgrid.Textgrid()
    
    end_time = grid_word_tuples[-1][1]
    wordTier = textgrid.IntervalTier('words', grid_word_tuples, 0, end_time)
    phoneTier = textgrid.IntervalTier('phones', grid_phones_tuples, 0, end_time)

    tg.addTier(wordTier)
    tg.addTier(phoneTier)
    
    tg.save(f"{basename}.TextGrid", format="long_textgrid", includeBlankSpaces=False)
