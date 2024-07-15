#!/usr/bin/env python
"""Tests for `transtokenizers` package."""

import pytest

from transtokenizers.transtokenizers import get_dataset_iterator

def test_get_dataset_iterator_open_subtitels():
    # Test case 1: Testing with open_subtitles dataset
    dataset_name = "open_subtitles"
    source_language = "en"
    target_language = "nl"
    iterator = get_dataset_iterator(dataset_name, source_language, target_language)

    # we expect tuples (sentence in source language, sentence in target language)
    # Check if tuples are returned
    assert isinstance(next(iterator), tuple)

    # Check if tuples contain two strings
    sentence1, sentence2 = next(iterator)
    assert isinstance(sentence1, str)
    assert isinstance(sentence2, str)

def test_get_dataset_iterator_nllb():
    # Test case 2: Testing with allenai/nllb dataset
    dataset_name = "allenai/nllb"
    source_language = "en"
    target_language = "nl"
    iterator = get_dataset_iterator(dataset_name, source_language, target_language)
    # we expect tuples (sentence in source language, sentence in target language)``
    # Check if tuples are returned
    assert isinstance(next(iterator), tuple)

    # Check if tuples contain two strings
    sentence1, sentence2 = next(iterator)
    assert isinstance(sentence1, str)
    assert isinstance(sentence2, str)

def test_get_dataset_iterator_nllb_non_latin():
    # Test case 3: Testing with allenai/nllb dataset with a non-latin script
    dataset_name = "allenai/nllb"
    source_language = "am"
    target_language = "en"
    iterator = get_dataset_iterator(dataset_name, source_language, target_language)
    # we expect tuples (sentence in source language, sentence in target language)``
    # Check if tuples are returned
    assert isinstance(next(iterator), tuple)

    # Check if tuples contain two strings
    sentence1, sentence2 = next(iterator)
    assert isinstance(sentence1, str)
    assert isinstance(sentence2, str)
