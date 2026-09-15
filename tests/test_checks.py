from rageval.questions.checks import has_context_reference, is_true, mentions_document_number, parse_json_object


def test_context_reference_english():
    assert has_context_reference("What does this report recommend?", "en")
    assert has_context_reference("According to the passage, who chaired the meeting?", "en")
    assert has_context_reference("Which body is mentioned above?", "en")
    assert not has_context_reference("What did the Security Council decide on small arms in 2004?", "en")


def test_context_reference_arabic_with_spelling_variants():
    assert has_context_reference("ما الذي يوصي به هذا التقرير؟", "ar")
    assert has_context_reference("ما هي الهيئة المذكوره اعلاه؟", "ar")  # unnormalized spelling
    assert not has_context_reference("ما الذي قررته الجمعية العامة بشأن الأسلحة الصغيرة؟", "ar")


def test_document_numbers_detected_in_both_scripts():
    assert mentions_document_number("What did A/47/10 recommend?", "en")
    assert mentions_document_number("What did resolution 47/33 request?", "en")
    assert mentions_document_number("ماذا طلب القرار ٤٧/٣٣؟", "ar")
    assert not mentions_document_number("What did the Assembly decide in 1992?", "en")


def test_parse_json_object_tolerates_fences():
    assert parse_json_object('```json\n{"question": "q"}\n```') == {"question": "q"}
    assert parse_json_object("no json here") is None
    assert parse_json_object("[1, 2]") is None
    assert parse_json_object("") is None


def test_is_true():
    assert is_true(True) and is_true("true") and not is_true("false") and not is_true(None)
