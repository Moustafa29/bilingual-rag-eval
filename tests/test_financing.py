"""Paragraphs are real financing-resolution text from the probe's sample, including its three errors."""

from rageval.corpus.financing import (
    arabic_confirms,
    canonical_mission,
    document_mission,
    extract_appropriation,
    parse_amount,
)

UNDOF_2001 = (
    "15. Decides to appropriate to the Special Account for the United Nations Disengagement Observer Force the amount of "
    "35,689,968 dollars gross (34,793,582 dollars net) for the maintenance of the Force for the period from 1 July 2001 to "
    "30 June 2002, inclusive of the amount of 1,044,551 dollars gross (916,696 dollars net) for the support account"
)
UNDOF_2001_AR = (
    "15 - تُقرر أن تعتمد للحساب الخاص لقوة الأمم المتحدة لمراقبة فض الاشتباك مبلغا إجماليه 35 689 968 دولارا "
    "(صافيه 34 793 582 دولارا) للإنفاق على القوة في الفترة مـــــن 1 تمـــــوز/يوليــــه 2001 إلى 30 حزيران/يونيه 2002"
)
UNTAET_2000 = (
    "14. Decides to appropriate to the Special Account for the United Nations Transitional Administration in East Timor the "
    "amount of 563 million dollars gross (546,051,600 dollars net) for the operation of the Transitional Administration for "
    "the period from 1 July 2000 to 30 June 2001, inclusive of the amount of 292,069,000 dollars gross (283,688,500 dollars net) "
    "authorized by the General Assembly in its resolution 54/246"
)
UNTAET_2000_AR = (
    "14 - تقرر أن تعتمد للحساب الخاص لإدارة الأمم المتحدة الانتقالية في تيمور الشرقية مبلغا إجماليه 563 مليون دولار "
    "(صافيه 546 051 600 دولار) لتشغيل الإدارة الانتقالية واستمرارها في الفترة من 1 تموز/يوليه 2000 إلى 30 حزيران/يونيه 2001"
)
MALI_2014 = (
    "20. Decides to appropriate to the Special Account the amount of 602 million dollars for the maintenance of the Mission for "
    "the period from 1 July 2013 to 30 June 2014, inclusive of the amount of 366,774,500 dollars previously authorized for the "
    "Mission for the period from 1 July to 31 December 2013 under the terms of its resolution 67/286;"
)
UNMISET_APPORTION = (
    "13. Decides also to apportion among Member States the amount of 80,096,775 dollars for the period from 1 January to "
    "30 June 2002, representing the balance of the appropriation for the period from 1 July 2001 to 30 June 2002 that has not "
    "been apportioned (53 million dollars)"
)
UNFICYP_1995 = (
    "11. Decides further to appropriate to the Special Account for the United Nations Peace-keeping Force in Cyprus the amount "
    "of 43,472,300 dollars gross (42,645,700 dollars net) for the period from 1 July 1995 to 30 June 1996, inclusive of the "
    "one-third share of the cost of the Force to be met through voluntary contributions"
)
UNFICYP_1995_AR = "١١ - تقرر كذلك أن تعتمد للحساب الخاص لقوة اﻷمــم المتحــدة لحفظ السلم في قبرص مبلغا إجماليه ٤٣ ٤٧٢ ٣٠٠ دوﻻر"


def test_grouped_amount_and_period_before_inclusive_clause():
    record = extract_appropriation(UNDOF_2001)
    assert record["amount_usd"] == 35_689_968
    assert record["period"] == "1 July 2001 - 30 June 2002"
    assert record["kind"] == "appropriation"
    assert record["mission_in_paragraph"] == "United Nations Disengagement Observer Force"


def test_amount_in_words_is_not_replaced_by_a_quoted_earlier_figure():
    # First probe error: took 292,069,000 (an earlier authorization) instead of 563 million.
    assert extract_appropriation(UNTAET_2000)["amount_usd"] == 563_000_000
    mali = extract_appropriation(MALI_2014)
    assert mali["amount_usd"] == 602_000_000
    assert mali["period"] == "1 July 2013 - 30 June 2014"
    assert mali["mission_in_paragraph"] is None  # "Special Account the amount": name comes from the document


def test_apportionment_is_not_an_appropriation():
    # First probe error: matched "appropriation" and a period quoted later in the paragraph.
    assert extract_appropriation(UNMISET_APPORTION) is None


SUPPORT_ACCOUNT_2001 = (
    "24. Decides also to appropriate to the Special Account for the Mission the amount of 8,260,509 dollars gross "
    "(7,249,409 dollars net) for the support account for peacekeeping operations and the amount of 862,915 dollars gross "
    "(774,893 dollars net) for the United Nations Logistics Base for the period from 1 July 2001 to 30 June 2002, to be "
    "apportioned among Member States"
)
UNMIBH_PARTIAL_1996 = (
    "7. Decides to appropriate the amount of 75,619,800 dollars gross (72,225,600 dollars net) for the maintenance of the "
    "Mission for the period from 1 July 1996 to 30 June 1997, inclusive of the amount of 1,918,300 dollars for the support "
    "account for peacekeeping operations, in addition to the amount of 75,619,800 dollars gross (72,225,600 dollars net) "
    "already appropriated for the period from 1 July to 31 December 1996 under the provisions of General"
)
UNFICYP_1998 = (
    "11. Decides to appropriate to the Special Account for the United Nations Peacekeeping Force in Cyprus an amount of "
    "45,276,160 dollars gross (43,536,860 dollars net) for the maintenance of the Force for the period from 1 July 1998 to "
    "30 June 1999, inclusive of an amount of 2,267,160 dollars for the support account for peacekeeping operations;"
)


def test_support_account_share_is_not_the_missions_appropriation():
    # Second probe sample: 8,260,509 dollars is a support-account share, and "the Mission" names no mission.
    record = extract_appropriation(SUPPORT_ACCOUNT_2001)
    assert record["kind"] == "support_account"
    assert record["mission_in_paragraph"] is None


def test_partial_year_appropriation_is_labelled():
    # Second probe sample: the second half of a year, in addition to an amount already appropriated.
    assert extract_appropriation(UNMIBH_PARTIAL_1996)["kind"] == "partial"


def test_support_account_mentioned_only_inside_the_inclusive_clause_stays_an_appropriation():
    record = extract_appropriation(UNFICYP_1998)
    assert record["kind"] == "appropriation"
    assert record["amount_usd"] == 45_276_160
    assert record["mission_in_paragraph"] == "United Nations Peacekeeping Force in Cyprus"


def test_additional_appropriation_is_labelled():
    text = UNDOF_2001.replace("the amount of 35,689,968", "an additional amount of 35,689,968")
    assert extract_appropriation(text)["kind"] == "additional"


def test_mission_names_are_canonicalised_across_spelling_variants():
    record = extract_appropriation(UNFICYP_1995)
    assert canonical_mission(record["mission_in_paragraph"]) == canonical_mission("United Nations Peacekeeping Force in Cyprus")
    assert canonical_mission("United Nations Mission in Liberia (continued)") == "united nations mission in liberia"


def test_document_mission_prefers_the_full_repeated_name():
    text = (
        "Financing of the United Nations Multidimensional Integrated Stabilization Mission in Mali "
        "The General Assembly, Having considered the report on the financing of the United Nations Multidimensional "
        "Integrated Stabilization Mission in Mali and the related report ... " + MALI_2014
    )
    assert document_mission(text) == "United Nations Multidimensional Integrated Stabilization Mission in Mali"
    assert document_mission("nothing relevant here") is None


def test_parse_amount_forms():
    assert parse_amount("585,682,100") == 585_682_100
    assert parse_amount("563 million") == 563_000_000
    assert parse_amount("6.5 million") == 6_500_000
    assert parse_amount("1.2 billion") == 1_200_000_000


def test_arabic_confirmation_for_grouped_and_word_amounts():
    assert arabic_confirms([UNDOF_2001_AR], "35,689,968")
    assert arabic_confirms([UNTAET_2000_AR], "563 million")
    assert arabic_confirms([UNFICYP_1995_AR], "43,472,300")  # Arabic-Indic digits
    assert not arabic_confirms([UNDOF_2001_AR], "35,689,969")
    assert not arabic_confirms(["نص لا يذكر أي مبلغ 35 689 968"], "36,000,000")
