import emphases


###############################################################################
# Test core.py
###############################################################################


def test_from_audio(text, audio_and_sample_rate):
    """Test emphases.from_text_and_audio"""
    audio, sample_rate = audio_and_sample_rate

    # Detect emphases
    alignment, results = emphases.from_text_and_audio(text, audio, sample_rate)

    # "sitting" and "different" should be emphasized
    for word, result in zip(alignment.words(), results):
        if str(word).lower() in ['sitting', 'different']:
            assert result
        else:
            assert not result
