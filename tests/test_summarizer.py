"""Tests for CX_DB8 core summarization logic."""

import numpy as np
import pytest

from cx_db8.embeddings import Embedder
from cx_db8.summarizer import (
    Granularity,
    ScoredSpan,
    SummaryResult,
    _create_word_ngrams,
    _segment_paragraphs,
    _segment_sentences,
    summarize,
)


# --- Unit tests (no model required) ---


class TestSegmentation:
    def test_segment_sentences_basic(self):
        text = "Hello world. This is a test. Another sentence here."
        sentences = _segment_sentences(text)
        assert len(sentences) >= 2
        assert any("Hello" in s for s in sentences)

    def test_segment_sentences_empty(self):
        assert _segment_sentences("") == []

    def test_segment_paragraphs_basic(self):
        text = "First paragraph here.\n\nSecond paragraph here."
        paragraphs = _segment_paragraphs(text)
        assert len(paragraphs) >= 1

    def test_segment_paragraphs_empty(self):
        assert _segment_paragraphs("") == []


class TestWordNgrams:
    def test_ngram_basic(self):
        words = ["the", "quick", "brown", "fox", "jumps"]
        ngrams = _create_word_ngrams(words, window_size=2)
        assert len(ngrams) == len(words)
        # First word has limited left context
        assert "the" in ngrams[0]
        # Middle word has full context
        assert "quick" in ngrams[2]
        assert "brown" in ngrams[2]
        assert "fox" in ngrams[2]

    def test_ngram_window_1(self):
        words = ["a", "b", "c", "d"]
        ngrams = _create_word_ngrams(words, window_size=1)
        assert len(ngrams) == 4
        assert ngrams[0] == "a b"  # [0:2]
        assert ngrams[1] == "a b c"  # [0:3]

    def test_ngram_single_word(self):
        ngrams = _create_word_ngrams(["hello"], window_size=5)
        assert len(ngrams) == 1
        assert "hello" in ngrams[0]


class TestSummaryResult:
    def test_compression_ratio(self):
        result = SummaryResult(
            card_text="test",
            query="test",
            granularity=Granularity.WORD,
            spans=[
                ScoredSpan("a", 0.9),
                ScoredSpan("b", 0.5),
                ScoredSpan("c", 0.1),
            ],
            underline_percentile=30.0,
            highlight_percentile=70.0,
            underline_threshold=0.3,
            highlight_threshold=0.7,
        )
        assert result.compression_ratio == pytest.approx(2 / 3)
        assert len(result.highlighted) == 1
        assert len(result.underlined) == 1
        assert len(result.removed) == 1

    def test_empty_spans(self):
        result = SummaryResult(
            card_text="",
            query="",
            granularity=Granularity.SENTENCE,
            spans=[],
            underline_percentile=50,
            highlight_percentile=80,
        )
        assert result.compression_ratio == 0.0


# --- Integration tests (require model download) ---


@pytest.fixture(scope="module")
def embedder():
    """Shared embedder instance (downloads model once)."""
    return Embedder("all-MiniLM-L6-v2")


class TestEmbedder:
    def test_encode_returns_array(self, embedder):
        embs = embedder.encode(["hello world"])
        assert isinstance(embs, np.ndarray)
        assert embs.shape[0] == 1
        assert embs.shape[1] > 0

    def test_similarity_shape(self, embedder):
        sims = embedder.similarity("test query", ["sentence one", "sentence two"])
        assert sims.shape == (2,)
        assert all(-1 <= s <= 1 for s in sims)

    def test_similar_texts_score_higher(self, embedder):
        sims = embedder.similarity(
            "the cat sat on the mat",
            ["a feline rested on the rug", "quantum physics is complex"],
        )
        assert sims[0] > sims[1]


class TestSummarize:
    def test_sentence_summarization(self, embedder):
        text = (
            "The SETI program searches for alien signals. "
            "Jerry Ehman discovered the Wow signal in 1977. "
            "The weather today is sunny and warm. "
            "Radio telescopes scan the sky for extraterrestrial communication."
        )
        result = summarize(
            card_text=text,
            query="SETI alien signal detection",
            embedder=embedder,
            granularity=Granularity.SENTENCE,
            underline_percentile=40,
            highlight_percentile=70,
        )
        assert len(result.spans) >= 3
        assert result.highlight_threshold >= result.underline_threshold
        # SETI-related sentences should score higher than weather
        seti_scores = [s.score for s in result.spans if "SETI" in s.text or "signal" in s.text]
        weather_scores = [s.score for s in result.spans if "weather" in s.text]
        if seti_scores and weather_scores:
            assert max(seti_scores) > max(weather_scores)

    def test_word_summarization(self, embedder):
        text = "The quick brown fox jumps over the lazy dog near the river"
        result = summarize(
            card_text=text,
            query="animals jumping",
            embedder=embedder,
            granularity=Granularity.WORD,
            underline_percentile=50,
            highlight_percentile=80,
            word_window_size=3,
        )
        assert len(result.spans) == len(text.split())

    def test_paragraph_summarization(self, embedder):
        text = "First paragraph about science.\n\nSecond about cooking.\n\nThird about science again."
        result = summarize(
            card_text=text,
            query="science",
            embedder=embedder,
            granularity=Granularity.PARAGRAPH,
        )
        assert len(result.spans) >= 1

    def test_generic_summary(self, embedder):
        text = "This is a test document about testing."
        result = summarize(
            card_text=text,
            query="",  # generic
            embedder=embedder,
        )
        assert len(result.spans) >= 1

    def test_embeddings_returned_when_requested(self, embedder):
        text = "Hello world. Goodbye world."
        result = summarize(
            card_text=text,
            query="greeting",
            embedder=embedder,
            want_embeddings=True,
        )
        assert result.embeddings is not None
        assert result.embeddings.shape[0] == len(result.spans)
