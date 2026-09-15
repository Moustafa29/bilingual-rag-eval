from rageval.corpus.symbols import doc_key, find_references, symbol_key


def test_find_references_symbols_and_ga_shorthand():
    text = (
        "See the report (A/47/10) and letter S/2004/567. The HIV/AIDS programme and "
        "ENGLISH/SPANISH originals are not symbols. In resolution 47/33 the Assembly acted; "
        "Human Rights Council resolution 7/19 is a different body."
    )
    assert find_references(text) == {"A/47/10", "S/2004/567", "A/RES/47/33"}


def test_find_references_strips_sentence_final_period():
    assert find_references("as requested in DP/2002/13/Add.1.") == {"DP/2002/13/Add.1"}


def test_security_council_references():
    text = "Recalling its resolution 1325 (2000) and S/RES/1711(2006), and the report (S/2006/123)."
    assert find_references(text) == {"S/RES/1325(2000)", "S/RES/1711(2006)", "S/2006/123"}


def test_joint_symbols_are_split_but_hyphenated_series_are_not():
    text = "letters dated 29 September 2000 (A/55/432-S/2000/921) to 4 August 2014 (A/ES-10/648-S/2014/567)"
    assert find_references(text) == {"A/55/432", "S/2000/921", "A/ES-10/648", "S/2014/567"}


def test_symbol_key_matches_corpus_paths():
    assert symbol_key("A/CN.4/452") == doc_key("1992/a/cn_4/452")
    assert symbol_key("DP/2002/13/Add.1") == doc_key("2002/dp/2002/13/add_1")
    # Each "." and space becomes its own underscore, as in the corpus folder "journal_no__2002".
    assert symbol_key("Journal No. 2002/1") == "journal_no__2002/1"
    # Parentheses and hyphens too; these cases come from real mismatches in the first build.
    assert symbol_key("S/RES/1711(2006)") == doc_key("2006/s/res/1711_2006_")
    assert symbol_key("A/ES-10/348") == doc_key("2006/a/es_10/348")
    assert symbol_key("S/AC.37/2003/(1455)/1") == doc_key("2003/s/ac_37/2003/_1455_/1")
    assert symbol_key("A/62/6(SECT.13)/ADD.1") == doc_key("2007/a/62/6_sect_13_/add_1")
