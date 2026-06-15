"""Discover hidden topics in news headlines with Latent Dirichlet Allocation.

This is a small, self-contained pipeline that:

  1. loads a corpus of news headlines,
  2. cleans the text (tokenize -> drop stopwords -> lemmatize),
  3. learns topics with two flavours of LDA (bag-of-words and TF-IDF), and
  4. renders the topics as word clouds and bar charts.

Run it end to end with::

    python src/topic_model.py --data data/sample_headlines.csv

See ``data/README.md`` for how to grab the full ~1.2M-headline dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gensim
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from gensim import corpora, models
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# Words that carry little topical meaning on their own. gensim ships a decent
# default list; we extend it with a few headline-specific fillers.
STOPWORDS = gensim.parsing.preprocessing.STOPWORDS.union(
    {"says", "say", "new", "year", "australia", "australian", "abc"}
)

_lemmatizer = WordNetLemmatizer()


def ensure_nltk() -> None:
    """Download the WordNet data the lemmatizer needs (no-op if cached)."""
    for pkg in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def preprocess(text: str) -> list[str]:
    """Turn a raw headline into a list of clean, meaningful tokens."""
    tokens = []
    for token in simple_preprocess(text, deacc=True):
        if token in STOPWORDS or len(token) < 4:
            continue
        tokens.append(_lemmatizer.lemmatize(token, pos="v"))
    return tokens


def load_headlines(path: Path, sample: int | None) -> pd.Series:
    """Read the CSV and return a cleaned Series of token lists."""
    frame = pd.read_csv(path, on_bad_lines="skip")
    headlines = frame["headline_text"].dropna()
    if sample and sample < len(headlines):
        headlines = headlines.sample(sample, random_state=42)
    print(f"Loaded {len(headlines):,} headlines from {path}")
    return headlines.map(preprocess)


def build_corpus(docs: pd.Series) -> tuple[corpora.Dictionary, list, list]:
    """Build the gensim dictionary plus bag-of-words and TF-IDF corpora."""
    dictionary = corpora.Dictionary(docs)
    # Drop words that are too rare (noise) or too common (no signal).
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100_000)

    bow_corpus = [dictionary.doc2bow(doc) for doc in docs]
    tfidf = models.TfidfModel(bow_corpus)
    tfidf_corpus = tfidf[bow_corpus]
    print(f"Vocabulary size after filtering: {len(dictionary):,} words")
    return dictionary, bow_corpus, tfidf_corpus


def train(corpus, dictionary, num_topics: int, passes: int):
    """Train an LDA model and print its topics."""
    model = models.LdaMulticore(
        corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=passes,
        workers=2,
        random_state=42,
    )
    for idx, topic in model.print_topics(-1):
        print(f"  Topic {idx:>2}: {topic}")
    return model


def save_wordclouds(model, num_topics: int, out: Path, title: str) -> None:
    """Render each topic as a word cloud in a single grid figure."""
    cols = 3
    rows = (num_topics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    for ax in axes.flatten():
        ax.axis("off")
    for topic_id in range(num_topics):
        weights = dict(model.show_topic(topic_id, topn=25))
        cloud = WordCloud(
            background_color="white",
            colormap="viridis",
            width=400,
            height=300,
            prefer_horizontal=0.9,
        ).generate_from_frequencies(weights)
        ax = axes.flatten()[topic_id]
        ax.imshow(cloud, interpolation="bilinear")
        ax.set_title(f"Topic {topic_id}", fontsize=11)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def save_top_words(model, out: Path, title: str, topics=(0, 1, 2, 3)) -> None:
    """Render horizontal bar charts of the top words for a few topics."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, topic_id in zip(axes.flatten(), topics):
        words, weights = zip(*model.show_topic(topic_id, topn=10))
        ax.barh(range(len(words)), weights, color="#4c72b0")
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_title(f"Topic {topic_id}")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/sample_headlines.csv"))
    parser.add_argument("--sample", type=int, default=None,
                        help="Randomly subsample this many headlines (default: all).")
    parser.add_argument("--num-topics", type=int, default=10)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    ensure_nltk()
    docs = load_headlines(args.data, args.sample)
    dictionary, bow_corpus, tfidf_corpus = build_corpus(docs)

    print("\nBag-of-Words topics:")
    bow_model = train(bow_corpus, dictionary, args.num_topics, args.passes)

    print("\nTF-IDF topics:")
    tfidf_model = train(tfidf_corpus, dictionary, args.num_topics, args.passes)

    args.figures.mkdir(parents=True, exist_ok=True)
    save_wordclouds(bow_model, args.num_topics,
                    args.figures / "topics_wordclouds.png",
                    "Topics discovered in news headlines (Bag-of-Words LDA)")
    save_top_words(bow_model, args.figures / "topics_top_words.png",
                   "Top words per topic (Bag-of-Words LDA)")


if __name__ == "__main__":
    main()
