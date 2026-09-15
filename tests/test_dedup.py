from rageval.questions.dedup import near_duplicate_groups, shingles

BASE = "The General Assembly requests the Secretary-General to report on the implementation of the present resolution at its sixtieth session"


def test_shingles_short_text_is_one_shingle():
    assert shingles("Security Council", 5) == {"security council"}
    assert shingles("", 5) == set()


def test_near_duplicates_grouped_and_different_text_not():
    chunks = {
        "gold": {"en": BASE},
        "copy": {"en": BASE.replace("sixtieth", "sixtieth")},
        "almost": {"en": BASE + " and thereafter annually"},
        "other": {"en": "The Security Council decided to extend the mandate of the mission for six months"},
    }
    groups = near_duplicate_groups({"gold"}, chunks, threshold=0.8)
    assert groups["gold"][0] == "gold"
    assert set(groups["gold"]) == {"gold", "copy", "almost"}
