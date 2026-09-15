import pytest

from rageval.corpus.tei import doc_id_from_path, iter_link_groups, parse_tei, parse_xtargets

TEI = """<?xml version="1.0" encoding="utf-8"?>
<TEI.2><teiHeader><fileDesc><publicationStmt>
<date>19920510</date>
<idno type="symbol">A/CN.4/452</idno>
<availability><p>Disclaimer paragraph that is not part of the body.</p></availability>
</publicationStmt></fileDesc></teiHeader>
<text><body>
  <p id="1"><s id="1:1" lang="en">Distr.</s></p>
  <p id="2">
    <s id="2:1" lang="en">First   sentence
      of paragraph two.</s>
    <s id="2:2" lang="en">Second &quot;sentence&quot;.</s>
  </p>
</body></text></TEI.2>""".encode("utf-8")


def test_parse_tei_reads_symbol_and_body_sentences_only():
    doc = parse_tei(TEI)
    assert doc.symbol == "A/CN.4/452"
    assert doc.date == "19920510"
    assert [(s.sid, s.pid) for s in doc.sentences] == [("1:1", "1"), ("2:1", "2"), ("2:2", "2")]
    assert doc.sentences[1].text == "First sentence of paragraph two."
    assert doc.sentences[2].text == 'Second "sentence".'


def test_parse_xtargets():
    link = parse_xtargets("34:1;31:1 31:2")
    assert link.ar == ("34:1",) and link.en == ("31:1", "31:2")
    assert not link.one_to_one
    unaligned = parse_xtargets("1:1;")
    assert unaligned.ar == ("1:1",) and unaligned.en == ()
    assert parse_xtargets("2:1;1:1").one_to_one


def test_doc_id_from_path():
    assert doc_id_from_path("ar/1992/a/cn_4/452.xml.gz", "ar") == "1992/a/cn_4/452"
    with pytest.raises(ValueError):
        doc_id_from_path("en/1992/a.xml.gz", "ar")


def test_iter_link_groups():
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<cesAlign version=\"1.0\">",
        '<linkGrp fromDoc="ar/1992/a/1.xml.gz" toDoc="en/1992/a/1.xml.gz" score="0.38">',
        '  <link type="1-0" xtargets="1:1;" score="0" />',
        '  <link type="1-1" xtargets="2:1;1:1" score="1" />',
        '  <link type="1-2" xtargets="3:1;2:1 2:2" score="0.7" />',
        "</linkGrp>",
        '<linkGrp fromDoc="ar/1993/s/2.xml.gz" toDoc="en/1993/s/2.xml.gz" score="0.5">',
        '  <link type="1-1" xtargets="1:1;1:1" score="1" />',
        "</linkGrp>",
        "</cesAlign>",
    ]
    groups = list(iter_link_groups(lines))
    assert len(groups) == 2
    first = groups[0]
    assert first.ar_path == "ar/1992/a/1.xml.gz" and first.score == 0.38
    assert [(l.ar, l.en) for l in first.links] == [(("1:1",), ()), (("2:1",), ("1:1",)), (("3:1",), ("2:1", "2:2"))]
    assert len(groups[1].links) == 1

    kept = list(iter_link_groups(lines, keep=lambda ar, en: en.startswith("en/1993/")))
    assert [g.en_path for g in kept] == ["en/1993/s/2.xml.gz"]
    assert len(kept[0].links) == 1
