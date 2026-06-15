# LDA Topic Modeling on News Headlines

An exploration of **Latent Dirichlet Allocation (LDA)** — a generative probabilistic model that discovers abstract "topics" hidden within a collection of documents. This project trains and compares two LDA variants on a real-world news dataset.

## What is LDA?

LDA is an unsupervised machine learning technique from the field of Natural Language Processing (NLP). Given a corpus of text, it assumes each document is a mixture of topics, and each topic is a mixture of words. The model learns these distributions automatically — no labels required.

It was introduced by Blei, Ng, and Jordan in 2003 and remains one of the foundational algorithms in topic modeling.

## Dataset

**ABC News Headlines** — a large collection of news headlines published by ABC News over nearly two decades. Each record contains a publish date and headline text.

## Models

Two LDA models are trained and compared:

| Model | Representation | Topics |
|-------|---------------|--------|
| LDA (Bag-of-Words) | Word frequency counts | 10 |
| LDA (TF-IDF) | Term frequency–inverse document frequency | 20 |

TF-IDF down-weights common words across the corpus, giving the second model a sharper signal on distinctive terms.

## Pipeline

1. **Preprocessing** — tokenization, stopword removal, and lemmatization + stemming using NLTK and Gensim
2. **N-gram detection** — bigrams and trigrams are identified to capture multi-word phrases (e.g., "prime_minister")
3. **Dictionary & corpus construction** — words filtered by frequency extremes to reduce noise
4. **Model training** — `LdaMulticore` with multiple passes for convergence
5. **Inference** — unseen documents are mapped to the learned topic distribution

## Tech Stack

- [Gensim](https://radimrehurek.com/gensim/) — LDA model training and corpus utilities
- [NLTK](https://www.nltk.org/) — lemmatization and stemming
- [Pandas](https://pandas.pydata.org/) — data loading and preprocessing

## Usage

```bash
git clone https://github.com/nishilfaldu/lda_models.git
cd lda_models/lda_news_model

pip install gensim pandas nltk

python lda_news_dataset.py
```
