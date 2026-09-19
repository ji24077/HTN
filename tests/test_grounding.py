import unicodedata
from dataclasses import asdict

import pytest

from gpushare.agent.grounding import check_grounding
from gpushare.agent.task import Record


def record(**changes):
    return Record(name="Ann Vale", age=19, org="Civic Lab", role="data steward", year=2019).model_copy(
        update=changes
    )


def test_supported_fields_have_original_offsets_and_all_occurrences():
    sentence = "Ann Vale, 19, joined Civic Lab in 2019 as data steward. Ann Vale signed."
    result = check_grounding(sentence, record())
    assert result.all_fields_present
    assert result.missing_fields == ()
    assert len(result.spans["name"]) == 2
    assert result.spans["name"][0].start == 0
    assert result.spans["name"][0].end == len("Ann Vale")
    for spans in result.spans.values():
        for span in spans:
            assert sentence[span.start : span.end] == span.text
    assert "semantic binding is not checked" in asdict(result)["scope"]


def test_missing_values_are_reported_without_repairing_prediction():
    prediction = record(name="Anne Vale", org="CL", role="manager")
    original = prediction.model_dump()
    result = check_grounding("Ann Vale, 19, joined Civic Lab in 2019 as data steward.", prediction)
    assert result.missing_fields == ("name", "org", "role")
    assert not result.all_fields_present
    assert prediction.model_dump() == original


def test_whitespace_case_and_original_span_are_preserved():
    sentence = "ANN\t\n VALE, 19, joined CIVIC   LAB in 2019 as DATA\nSTEWARD."
    result = check_grounding(sentence, record())
    assert result.all_fields_present
    assert result.spans["name"][0].text == "ANN\t\n VALE"
    assert result.spans["role"][0].text == "DATA\nSTEWARD"


@pytest.mark.parametrize("form", ["NFC", "NFD"])
def test_canonically_equivalent_unicode_keeps_original_span(form):
    name = unicodedata.normalize(form, "Inés Åström")
    sentence = f"{name}, 19, joined Civic Lab in 2019 as data steward."
    result = check_grounding(sentence, record(name="Inés Åström"))
    assert result.all_fields_present
    span = result.spans["name"][0]
    assert span.text == name
    assert span.start == 0 and span.end == len(name)


@pytest.mark.parametrize("predicted", ["Ines Åström", "Inés Astrom", "Ines Astrom"])
def test_accents_are_not_discarded(predicted):
    result = check_grounding("Inés Åström, 19, Civic Lab, data steward, 2019.", record(name=predicted))
    assert result.missing_fields == ("name",)


def test_casefold_expansion_retains_source_offsets():
    sentence = "Straße Vale, 19, Civic Lab, data steward, 2019."
    result = check_grounding(sentence, record(name="STRASSE VALE"))
    assert result.all_fields_present
    assert result.spans["name"][0].text == "Straße Vale"


def test_punctuation_braces_and_quoted_values_remain_literal():
    prediction = record(org='Test {Works} & "Co"', role="curator (maps), grade-II")
    sentence = 'Ann Vale, 19, joined Test {Works} & "Co" in 2019 as curator (maps), grade-II.'
    result = check_grounding(sentence, prediction)
    assert result.all_fields_present
    assert result.spans["org"][0].text == prediction.org
    assert result.spans["role"][0].text == prediction.role


@pytest.mark.parametrize("sentence", ["Only 2019 is recorded.", "The number is 190."])
def test_age_requires_digit_boundaries(sentence):
    assert "age" in check_grounding(sentence, record()).missing_fields


def test_numeric_quotes_are_allowed_and_year_is_not_a_prefix():
    result = check_grounding('Age: "19"; identifier: 20190.', record())
    assert result.spans["age"][0].text == "19"
    assert "year" in result.missing_fields


@pytest.mark.parametrize("source", ["Joanne", "Annabelle", "McAnn", "Ann_2"])
def test_text_values_require_word_boundaries(source):
    assert "name" in check_grounding(source, record(name="Ann")).missing_fields


def test_empty_text_and_empty_predicted_string_do_not_count_as_evidence():
    assert len(check_grounding("", record()).missing_fields) == 5
    assert "name" in check_grounding("anything", record(name="  ")).missing_fields


def test_present_founding_year_is_not_semantically_verified():
    sentence = "Civic Lab opened in 2019. Ann Vale, 19, joined it as data steward in 2023."
    incorrect_appointment = record(year=2019)
    result = check_grounding(sentence, incorrect_appointment)
    # Deliberately all present despite an incorrect appointment year. Downstream
    # consumers must never rename all_fields_present to factual_correctness.
    assert result.all_fields_present
    assert result.spans["year"][0].text == "2019"
    assert "semantic binding is not checked" in result.scope
