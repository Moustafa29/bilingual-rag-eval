from rageval.corpus.numbers import (
    audit_numbers,
    correct_reversed_numbers,
    english_numbers,
    has_thousands_number,
    is_numeric_question,
)


def test_english_numbers_with_thousands_separators_only():
    assert english_numbers("about 2,000 troops, $1,974,200, in 2003, ratio 3.5") == [["2", "000"], ["1", "974", "200"]]


def test_numbers_next_to_clause_and_list_commas_are_found():
    # Regression: the first pattern skipped any number with an adjacent comma.
    assert english_numbers("costs of 2,000, 5,000, and 12,500.") == [["2", "000"], ["5", "000"], ["12", "500"]]
    assert english_numbers("figure 1,974.5 million") == [["1", "974"]]
    assert english_numbers("code 12,3456") == []


def test_audit_classifies_arabic_forms():
    en = "278,707 persons"
    assert audit_numbers(en, "أن 707 278 أشخاص") == {"reversed_space": 1}
    assert audit_numbers(en, "أن 278 707 أشخاص") == {"forward_space": 1}
    assert audit_numbers(en, "أن 278,707 أشخاص") == {"comma": 1}
    assert audit_numbers(en, "أن 278707 أشخاص") == {"joined": 1}
    assert audit_numbers(en, "لا أرقام") == {"not_found": 1}
    assert audit_numbers("200,200", "200 200") == {"ambiguous": 1}


def test_audit_reads_arabic_indic_digits():
    assert audit_numbers("278,707", "٧٠٧ ٢٧٨") == {"reversed_space": 1}


def test_correction_reverses_groups_back():
    text, n = correct_reversed_numbers("278,707 persons", "أن 707 278 أشخاص")
    assert (text, n) == ("أن 278 707 أشخاص", 1)
    assert audit_numbers("278,707 persons", text) == {"forward_space": 1}


def test_correction_preserves_digit_script_and_separator():
    assert correct_reversed_numbers("278,707", "٧٠٧ ٢٧٨") == ("٢٧٨ ٧٠٧", 1)
    assert correct_reversed_numbers("50,000", "000 50 دولار") == ("50 000 دولار", 1)


def test_three_groups():
    assert correct_reversed_numbers("$1,974,200", "200 974 1 دولار") == ("1 974 200 دولار", 1)


def test_no_correction_without_the_english_number():
    assert correct_reversed_numbers("no numbers here", "000 50") == ("000 50", 0)


def test_no_partial_match_inside_a_longer_digit_run():
    # "000 2" is preceded by "3 ", so it is part of a different number and must stay untouched.
    assert correct_reversed_numbers("2,000", "3 000 2") == ("3 000 2", 0)
    assert correct_reversed_numbers("2,000", "000 2 5") == ("000 2 5", 0)


def test_palindromic_groups_are_left_alone():
    assert correct_reversed_numbers("200,200", "200 200") == ("200 200", 0)


def test_numeric_question_subset():
    assert has_thousands_number("2.9 million") is False
    assert has_thousands_number("278,707 persons") is True
    q = {"question": {"en": "How many persons held displaced status?"}, "answer": {"en": "278,707 persons"}}
    assert is_numeric_question(q)
    q["answer"]["en"] = "the Habitat Agenda"
    assert not is_numeric_question(q)
