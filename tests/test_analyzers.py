from rageval.retrieval.analyzers import analyze, stem_arabic


def test_raw_keeps_clitics_attached():
    assert analyze("والمنظمة", "ar", "raw") == ["والمنظمة"]


def test_norm_folds_orthography_but_does_not_stem():
    assert analyze("الأمم المتحدة", "ar", "norm") == ["الامم", "المتحده"]


def test_light_strips_conjunction_and_article_so_forms_match():
    # "and the organization" and "organization" reach the same stem only with light stemming.
    assert analyze("والمنظمة", "ar", "light") == analyze("منظمة", "ar", "light")
    assert analyze("والمنظمة", "ar", "raw") != analyze("منظمة", "ar", "raw")


def test_arabic_stemmer_prefix_rules():
    assert stem_arabic("الكتاب") == "كتاب"
    assert stem_arabic("بالكتاب") == "كتاب"
    assert stem_arabic("للمجلس") == "مجلس"
    # A one-letter "wa-" prefix is only stripped from words of 4+ letters.
    assert stem_arabic("وقت") == "وقت"
    assert stem_arabic("وكتب") == "كتب"


def test_arabic_stemmer_suffix_rules_leave_at_least_two_letters():
    assert stem_arabic("معلمون") == "معلم"
    assert stem_arabic("سيارات") == "سيار"
    assert stem_arabic("هو") == "هو"


def test_broken_plurals_are_a_known_limitation():
    # كتاب (book) and كتب (books): light stemming cannot relate an internal vowel change.
    assert stem_arabic("كتاب") != stem_arabic("كتب")


def test_english_light_stems():
    assert analyze("Resolutions recommended", "en", "light") == ["resolut", "recommend"]
    assert analyze("Resolutions", "en", "raw") == ["resolutions"]


def test_digits_fold_in_norm_not_raw():
    assert analyze("القرار ٤٧/٣٣", "ar", "norm") == ["القرار", "47", "33"]
    assert analyze("٤٧", "ar", "raw") == ["٤٧"]
