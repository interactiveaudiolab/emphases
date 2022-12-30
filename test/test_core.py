import emphases


###############################################################################
# Test core.py
###############################################################################


def test_from_audio(text, audio):
    """Test emphases.from_text_and_audio"""
    # Detect emphases
    alignment, results = emphases.from_text_and_audio(
        text,
        audio,
        emphases.SAMPLE_RATE)

    # "sitting" and "different" should be emphasized
    for word, result in zip(alignment.words(), results):
        if str(word).lower() in ['sitting', 'different']:
            assert result
        else:
            assert not result
