"""Core summarization logic for CX_DB8."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray
import syntok.segmenter as segmenter

from cx_db8.embeddings import Embedder


class Granularity(str, Enum):
    WORD = "word"
    PHRASE = "phrase"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"


@dataclass
class ScoredSpan:
    """A text span with its similarity score."""

    text: str
    score: float


@dataclass
class SummaryResult:
    """Full result of a summarization run."""

    card_text: str
    query: str
    granularity: Granularity
    spans: list[ScoredSpan]
    underline_percentile: float
    highlight_percentile: float
    underline_threshold: float = 0.0
    highlight_threshold: float = 0.0
    embeddings: NDArray[np.float32] | None = None

    @property
    def highlighted(self) -> list[ScoredSpan]:
        return [s for s in self.spans if s.score >= self.highlight_threshold]

    @property
    def underlined(self) -> list[ScoredSpan]:
        return [
            s
            for s in self.spans
            if self.underline_threshold <= s.score < self.highlight_threshold
        ]

    @property
    def removed(self) -> list[ScoredSpan]:
        return [s for s in self.spans if s.score < self.underline_threshold]

    @property
    def compression_ratio(self) -> float:
        total = len(self.spans)
        kept = len(self.highlighted) + len(self.underlined)
        return kept / total if total > 0 else 0.0


def _segment_sentences(text: str) -> list[str]:
    """Split text into sentences using syntok."""
    sentences = []
    for paragraph in segmenter.analyze(text):
        for sentence in paragraph:
            tokens = []
            for token in sentence:
                tokens.append(token.spacing + token.value)
            sentences.append("".join(tokens).strip())
    return [s for s in sentences if s]


def _segment_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs using syntok."""
    paragraphs = []
    for paragraph in segmenter.analyze(text):
        tokens = []
        for sentence in paragraph:
            for token in sentence:
                tokens.append(token.spacing + token.value)
        para_text = "".join(tokens).strip()
        if para_text:
            paragraphs.append(para_text)
    return paragraphs


def _create_word_ngrams(words: list[str], window_size: int) -> list[str]:
    """Create context-window n-grams around each word."""
    ngrams = []
    for i in range(len(words)):
        lo = max(0, i - window_size)
        hi = min(len(words), i + window_size + 1)
        ngram = " ".join(words[lo:hi])
        ngrams.append(ngram if ngram else " ")
    return ngrams


def _bridge_gaps(
    words: list[str],
    scores: NDArray[np.float64],
    ul_thresh: float,
    bridge_size: int = 3,
) -> list[ScoredSpan]:
    """Score words then bridge small gaps so underlined text reads grammatically.

    All words are kept (this is strictly extractive). After classifying each
    word as kept (above underline threshold) or not, any run of below-threshold
    words that is <= bridge_size tokens *and* sits between two kept words gets
    promoted to the underline threshold so that grammatical connectors
    (articles, prepositions, conjunctions) are preserved when reading only the
    underlined/highlighted portions.
    """
    n = len(words)
    promoted = np.array(scores, dtype=np.float64)

    # Identify which words are "kept" (at or above underline threshold)
    kept = promoted >= ul_thresh

    # Bridge small gaps between kept words
    i = 0
    while i < n:
        if not kept[i]:
            # Find extent of this below-threshold run
            j = i
            while j < n and not kept[j]:
                j += 1
            gap_len = j - i
            # Promote if bounded by kept words on both sides and small enough
            has_left = i > 0 and kept[i - 1]
            has_right = j < n and kept[j]
            if has_left and has_right and gap_len <= bridge_size:
                for k in range(i, j):
                    promoted[k] = ul_thresh  # promote to underline level
            i = j
        else:
            i += 1

    return [ScoredSpan(text=w, score=float(s)) for w, s in zip(words, promoted)]


def summarize(
    card_text: str,
    query: str,
    embedder: Embedder,
    granularity: Granularity = Granularity.SENTENCE,
    underline_percentile: float = 70.0,
    highlight_percentile: float = 85.0,
    word_window_size: int = 10,
    bridge_gap_size: int = 3,
    want_embeddings: bool = False,
) -> SummaryResult:
    """Run extractive summarization on card text.

    Args:
        card_text: The evidence/card text to summarize.
        query: The card tag / query to summarize in terms of.
                If empty, summarizes in terms of the card itself.
        embedder: The embedding engine to use.
        granularity: Word, phrase, sentence, or paragraph level.
        underline_percentile: Percentile threshold for underlining (0-100).
        highlight_percentile: Percentile threshold for highlighting (0-100).
        word_window_size: Bi-directional context window for word/phrase n-grams.
        bridge_gap_size: Max removed words to bridge in phrase mode (default 3).
        want_embeddings: If True, return the raw embeddings for visualization.

    Returns:
        A SummaryResult with scored spans and thresholds.
    """
    effective_query = query if query else card_text

    if granularity in (Granularity.WORD, Granularity.PHRASE):
        words = card_text.split()
        ngrams = _create_word_ngrams(words, word_window_size)
        if want_embeddings:
            scores, embs = embedder.encode_with_embeddings(effective_query, ngrams)
        else:
            scores = embedder.similarity(effective_query, ngrams)
            embs = None

        # Compute thresholds first (on raw word scores)
        all_scores_list = [float(s) for s in scores]
        if len(all_scores_list) < 2:
            ul_thresh = hl_thresh = 0.0
        else:
            ul_thresh = float(np.percentile(all_scores_list, underline_percentile))
            hl_thresh = float(np.percentile(all_scores_list, highlight_percentile))

        if granularity == Granularity.PHRASE:
            spans = _bridge_gaps(words, scores, ul_thresh, bridge_gap_size)
        else:
            spans = [ScoredSpan(text=w, score=float(s)) for w, s in zip(words, scores)]

        return SummaryResult(
            card_text=card_text,
            query=effective_query,
            granularity=granularity,
            spans=spans,
            underline_percentile=underline_percentile,
            highlight_percentile=highlight_percentile,
            underline_threshold=ul_thresh,
            highlight_threshold=hl_thresh,
            embeddings=embs,
        )

    elif granularity == Granularity.SENTENCE:
        sentences = _segment_sentences(card_text)
        if not sentences:
            sentences = [card_text]
        if want_embeddings:
            scores, embs = embedder.encode_with_embeddings(effective_query, sentences)
        else:
            scores = embedder.similarity(effective_query, sentences)
            embs = None
        spans = [ScoredSpan(text=s, score=float(sc)) for s, sc in zip(sentences, scores)]

    elif granularity == Granularity.PARAGRAPH:
        paragraphs = _segment_paragraphs(card_text)
        if not paragraphs:
            paragraphs = [card_text]
        if want_embeddings:
            scores, embs = embedder.encode_with_embeddings(effective_query, paragraphs)
        else:
            scores = embedder.similarity(effective_query, paragraphs)
            embs = None
        spans = [ScoredSpan(text=p, score=float(sc)) for p, sc in zip(paragraphs, scores)]
    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    all_scores = [s.score for s in spans]
    if len(all_scores) < 2:
        ul_thresh = hl_thresh = 0.0
    else:
        ul_thresh = float(np.percentile(all_scores, underline_percentile))
        hl_thresh = float(np.percentile(all_scores, highlight_percentile))

    return SummaryResult(
        card_text=card_text,
        query=effective_query,
        granularity=granularity,
        spans=spans,
        underline_percentile=underline_percentile,
        highlight_percentile=highlight_percentile,
        underline_threshold=ul_thresh,
        highlight_threshold=hl_thresh,
        embeddings=embs,
    )
