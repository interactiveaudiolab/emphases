
def read_textgrid(filename, sample_rate=200):
    import tgt
    tg = tgt.read_textgrid(filename)

    tiers = []
    labs = {}

    for tier in tg.get_tier_names():
        if (tg.get_tier_by_name(tier)).tier_type()!='IntervalTier':
            continue
        tiers.append(tg.get_tier_by_name(tier))

        lab = []
        for a in tiers[-1].annotations:

            # this was for some past experiment
            if a.text in ["p1","p2","p3","p4","p5","p6","p7"]:
                lab[-1][-1]=lab[-1][-1]+"_"+a.text
            else:
                lab.append([a.start_time*sample_rate,a.end_time*sample_rate,a.text])
        labs[tier.lower()] = lab
    for i in range(len(labs['prosody'])):
        if labs['prosody'][i][2][-2:] not in ["p1","p2","p3","p4","p5","p6","p7"]:
            labs['prosody'][i][2]+="_p0"

    return labs


def htk_to_ms(htk_time):
    """
    Convert time in HTK (100 ns) units to 5 ms
    """
    if type(htk_time)==type("string"):
        htk_time = float(htk_time)
    return htk_time / 50000.0


def read_htk_label(fname, htk_time=True):
    """
    Read HTK label, assume: "start end phone word", where word is optional.
    Convert times from HTK units to MS
    """
    import codecs

    try:
        f = codecs.open(fname, 'r', 'utf-8')
    except:
        raise Exception("htk label file %s not found" % fname)

    label = f.readlines()
    f.close()

    label = [line.split() for line in label]

    segments = []
    words = []
    prev_end = 0.0
    prev_start = 0.0
    prev_word = '!SIL'
    word = ''
    for line in label:
        if len(line) == 4 and line[2] == 'skip':
            continue
        word = False
        if len(line)==3:
            (start,end,segment) = line
            if start == 'nan':
                continue

        elif len(line) == 4:

            start, end, segment, word = line

        else:

            continue

        if htk_time == True:
            end = htk_to_ms(int(end))
            start = htk_to_ms(int(start))
        else:
            # 5ms frame
            end = float(end)*200
            start = float(start)*200

        if start == end:
            continue

        prev_end = start

        segments.append([int(start), int(end), segment]) #

        """
        if word or segments[-1][2] in ["SIL", "pause", '#']:
            try:
                if prev_word not in ["!SIL", "pause"] and prev_word[0]!= "!" and prev_word[0]!="_"  and prev_word[0]!='#':
                    words.append([int(prev_start), int(prev_end),prev_word]) #, prev_word])
            except:
                pass
        """
        if word:
            words.append([int(prev_start), int(prev_end),prev_word])
            prev_start = start
            prev_word = word
            word = ''
    if len(label[-1]) == 4:
        words.append([htk_to_ms(float(label[-1][0])), htk_to_ms(float(label[-1][1])), label[-1][3]])
    labs = {}
    if len(words) > 0:
        labs['words'] = words
    labs['segments'] = segments

    return labs
