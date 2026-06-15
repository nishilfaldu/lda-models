# Data

## `sample_headlines.csv` (included)

A 25,000-headline random sample of the full dataset, committed so the project
runs the moment you clone it:

```bash
python src/topic_model.py --data data/sample_headlines.csv
```

It's enough to surface clear, sensible topics in well under a minute.

## Full dataset (optional, ~60 MB)

The complete corpus is **A Million News Headlines** — every headline published
by the Australian Broadcasting Corporation (ABC) over roughly two decades
(2003–2021), about **1.2 million** of them.

Download `abcnews-date-text.csv` into this folder from Kaggle:

- https://www.kaggle.com/datasets/therohk/million-headlines

Then point the script at it:

```bash
python src/topic_model.py --data data/abcnews-date-text.csv --sample 200000
```

The full file is intentionally **not** committed (it's in `.gitignore`) to keep
clones small. Each row is just two columns: `publish_date,headline_text`.
