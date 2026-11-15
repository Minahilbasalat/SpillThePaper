from src.data.text_quality import build_known_words, quality

CORPUS = ["Explain how TCP congestion control reacts to packet loss and delay in the network"] * 5 + \
         ["Calculate the total transmission delay of a packet over a link with given size"] * 5


def test_garbled_text_scores_low_and_real_text_high():
    known = build_known_words(CORPUS)
    real = "Calculate the total transmission delay of a packet over the link"
    junk = "Total tres delay f GONS sews t ARTT+ Sil trams mission size z noma rloms rosin"
    assert quality(real, known) > 0.9
    assert quality(junk, known) < 0.6


def test_short_text_is_not_judged():
    assert quality("Q1 (5)", set()) == 1.0
