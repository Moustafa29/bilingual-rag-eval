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


def test_in_the_document_is_a_context_reference_unless_it_describes_the_document():
    # Rejected single-hop #3 from the smoke run, in both languages.
    assert has_context_reference("من أي دولة ينتمي السيد شيفيش المذكور في الوثيقة؟", "ar")
    assert has_context_reference("From which State does Mr. Shafeesh mentioned in the document belong?", "en")
    assert has_context_reference("ما الرقم الوارد في النص؟", "ar")
    assert has_context_reference("According to the report, how many refugees are there?", "en")
    # Bridge #11's phrasing describes the document and must stay allowed.
    assert not has_context_reference("في الوثيقة التي طلبت الجمعية العامة من الأمين العام تقديم اقتراح شامل، أي مادة؟", "ar")
    assert not has_context_reference("In the document that the General Assembly requested, which rule is reaffirmed?", "en")
    assert not has_context_reference("According to the report of the Secretary-General on refugees, how many are there?", "en")


def test_the_mentioned_decision_is_a_context_reference():
    # Accepted single-hop #5 from the smoke run, which the earlier filter missed.
    assert has_context_reference("من الذي يُقترح أن يرأس الفريق العامل وفقاً للقرار المذكور؟", "ar")
    assert has_context_reference("Who is proposed to chair the working team according to the mentioned decision?", "en")
    assert has_context_reference("ما الذي تقرر في الاجتماع المذكور؟", "ar")
    assert not has_context_reference("ما الذي قرره مجلس الأمن في قراره بشأن الصومال؟", "ar")


def test_arabic_according_to_the_document_is_caught_on_the_source_side():
    # Arabic-source questions from the numeric run that were caught only after translation.
    assert has_context_reference("كم عدد اللاجئين الذين فروا إلى تشاد وفقًا للوثيقة؟", "ar")
    assert has_context_reference("ما هو مقدار التخفيض المالي الذي لاحظته المجموعة وفقًا للتقرير؟", "ar")
    assert has_context_reference("ما هو عدد الأشخاص المصابين بالمهق في بعض أنحاء أفريقيا حسب النص؟", "ar")
    assert has_context_reference("كم عدد الهكتارات التي ذكرها النص في مشروع إعادة التحريج؟", "ar")
    # Described documents stay allowed, and ordinary "according to" phrases are untouched.
    assert not has_context_reference("وفقًا للوثيقة التي قدمها الأمين العام، كم عدد اللاجئين؟", "ar")
    assert not has_context_reference("كم عدد اللاجئين وفقًا لتقديرات المفوضية؟", "ar")
    # Longer words that merely start with the noun are not references to an unseen passage.
    assert not has_context_reference("ما هي الدول الملزمة وفقًا لنصوص الاتفاقية؟", "ar")
    assert not has_context_reference("ما الحقوق التي تذكرها النصوص الدولية؟", "ar")


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
