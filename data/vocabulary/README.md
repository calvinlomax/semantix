# Vocabulary roles

`allowed_guesses.txt` contains 94,037 broad English guess candidates. It is an
input to streaming Google News model preparation, not a runtime allowlist.
The current reduced `.kv` retains 73,604 candidates with genuine embeddings;
20,433 requested forms are absent and listed in the model preparation manifest.
Only forms resolvable against the `.kv` can be played. Case normalization or
an unambiguous lemma can accept a submitted form absent from this text file.

`level_words.txt` contains 2,676 manually curated, familiar puzzle words, of
which 2,674 currently have embeddings (`doughnut` and `grey` are unavailable).
Puzzle starts/answers must come from this list and exist exactly in the model.
Search, hints, and landmarks use its small retained matrix. The original
`words.txt` remains the reviewed seed and preserves target ordering, so adding
guess embeddings does not invalidate existing calibrated levels.

`blocked.txt` is an additional runtime input denylist. Both the builder and
runtime reject blocked grammatical families, even when an exact inflection
has its own embedding. It is not a complete content audit.

## Rebuild

From the repository root, install `backend/requirements-preparation.txt`, run
`backend.scripts.prepare_nlp` once, then:

```sh
.venv/bin/python -m backend.scripts.build_large_vocab --size 100000
.venv/bin/python -m backend.scripts.prepare_model \
  --source data/models/word2vec-google-news-300.gz \
  --vocabulary data/vocabulary/allowed_guesses.txt \
  --output data/models/google-news-300.kv
```

The builder uses wordfreq's English frequency list and WordNet common lexical
senses/inflections. It retains lowercase ASCII tokens of 2–24 letters, rejects
punctuation and numbers, excludes blocked families and names without a common
lexical sense, deduplicates, and ranks by frequency with alphabetical ties.
Guess output is alphabetically ordered; target output preserves reviewed seed
order. A common sense remains usable when a word is also a name (such as
`rose`). Proper-name filtering is lexical evidence, not a guarantee about every
possible sense of every word.

The default 100,000 ceiling yields 94,037 eligible candidates from these source
versions; it never pads output to reach a quota. Recommended source size is
50,000–100,000, with a maximum of 150,000. The broader list never becomes the
target source. `build_manifest.json` records versions, counts, and SHA-256 hashes;
rerunning with the same inputs yields identical text and manifest output.

Commit these small text files and the build manifest. The separately prepared
model, source archive, and NLP downloads remain excluded by `.gitignore`.

## Sources and attribution

The generated guess list combines and filters source vocabulary; it does not
redistribute frequency tables. Its derived data is distributed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), with the upstream
WordNet terms retained. Preserve the comments and this attribution when
redistributing the generated list.

- **wordfreq 3.1.1**, by **Robyn Speer**: [project and credits](https://github.com/rspeer/wordfreq).
  The library is Apache-licensed; its frequency data is CC BY-SA 4.0.
- **WordNet 3.0**, **Princeton University**: [license and attribution](https://wordnet.princeton.edu/license-and-commercial-use).
  WordNet supplies lexical forms, grammatical analyses, and proper-name evidence.
- **SUBTLEX**, **Marc Brysbaert and collaborators**: [freely available subtitle frequency data](http://crr.ugent.be/programs-data/subtitle-frequencies),
  incorporated by wordfreq. See wordfreq's credits for the individual SUBTLEX publications.
- wordfreq also credits [Google Books Ngrams](https://books.google.com/ngrams),
  [OpenSubtitles](https://www.opensubtitles.org/), Wikipedia, the Leeds Internet
  Corpus, and ParaCrawl. Its linked documentation contains the full upstream credits.
- `words.txt` is Semantix's existing manually curated seed spanning nature,
  everyday objects, relationships, art, science, travel, and civic life.
