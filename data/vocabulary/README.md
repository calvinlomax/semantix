`words.txt` is a manually curated, familiar English allowlist spanning nature,
food, travel, everyday objects, relationships, art, science, and civic life.
It contains lowercase, single-word tokens suitable for varied semantic routes.
Only words actually found in Google's Google News embeddings become playable.
The preparation manifest records any missing words; no replacements are invented.

`blocked.txt` is a small additional input blocklist. An allowlist is more
predictable than attempting to filter all three million original news tokens.
This vocabulary is family-oriented but has not undergone a formal content audit.
To customize it, add one familiar word per line and re-run preparation and level
generation. A reduced model only accepts retained allowlist words.
