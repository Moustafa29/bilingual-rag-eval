from rageval.text import answer_tokens, contains_evidence, contains_span, normalize, token_f1


def test_arabic_normalization_removes_diacritics_and_unifies_alef():
    assert normalize("إِنَّ الأُمَمَ", "ar") == "ان الامم"


def test_arabic_presentation_form_ligature_is_folded():
    # "الأمم" written with the isolated lam-alef-hamza ligature U+FEF7, as in older UN files.
    assert normalize("اﻷمم", "ar") == normalize("الأمم", "ar") == "الامم"


def test_tatweel_teh_marbuta_and_alef_maksura():
    assert normalize("مقدمـــة", "ar") == "مقدمه"
    assert normalize("على", "ar") == "علي"


def test_arabic_indic_digits_fold_to_ascii():
    assert normalize("القرار ٤٧/٣٣", "ar") == "القرار 47/33"


def test_english_lowercase_and_whitespace():
    assert normalize("  The  General\nAssembly ", "en") == "the general assembly"


def test_answer_tokens_drop_articles():
    assert answer_tokens("the Security Council", "en") == ["security", "council"]
    assert answer_tokens("والجمعية العامة", "ar") == ["جمعيه", "عامه"]


def test_definite_article_not_stripped_from_short_words():
    # "الى" minus "ال" would leave one letter; it must stay intact.
    assert answer_tokens("الى", "ar") == ["الي"]


def test_token_f1():
    assert token_f1("the Security Council", "Security Council", "en") == 1.0
    assert token_f1("General Assembly", "Security Council", "en") == 0.0
    assert token_f1("الجمعية العامة", "جمعية عامة", "ar") == 1.0
    assert abs(token_f1("25 November 1992", "25 November", "en") - 0.8) < 1e-9


def test_token_f1_empty():
    assert token_f1("", "", "en") == 1.0
    assert token_f1("the", "Council", "en") == 0.0


def test_contains_span_respects_token_boundaries():
    assert contains_span("The Security Council met.", "security council", "en")
    assert not contains_span("The Security Council met.", "Counc", "en")
    assert not contains_span("anything", "", "en")


def test_evidence_tolerates_dropped_arabic_conjunction():
    # Bridge re-test #1: the generator copied "وإذ يؤكد ..." as "إذ يؤكد ...".
    passage = "وإذ يؤكد الحاجة إلى استجابة شاملة من المجتمع الدولي لمعالجة مشكلة القرصنة،"
    copy = "إذ يؤكد الحاجة إلى استجابة شاملة من المجتمع الدولي لمعالجة مشكلة القرصنة"
    assert not contains_span(passage, copy, "ar")
    assert contains_evidence(passage, copy, "ar")


def test_evidence_tolerates_hyphens_lost_in_corpus_text():
    # Bridge re-test #14: the passage says "SecretaryGeneral's", the copy "Secretary-General's".
    passage = "He supported the recommendations in paragraph 77 of the SecretaryGeneral's report on poverty."
    copy = "He supported the recommendations in paragraph 77 of the Secretary-General's report on poverty."
    assert not contains_span(passage, copy, "en")
    assert contains_evidence(passage, copy, "en")


def test_evidence_still_rejects_a_different_sentence():
    assert not contains_evidence("The Council decided to extend the mandate.", "The Assembly decided to end the mission.", "en")
    assert not contains_evidence("قررت الجمعية تمديد الولاية.", "قرر المجلس إنهاء البعثة.", "ar")
    assert not contains_evidence("anything", "", "en")


def test_contains_span_arabic_variants():
    assert contains_span("اعتمدت الجمعيَّة العامة القرار ٤٧/٣٣", "الجمعية العامة القرار 47/33", "ar")
