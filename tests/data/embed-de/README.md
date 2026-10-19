# embed-de – German retrieval test corpus (skilled trades)

## Purpose

Quality gate for embedding models (reference: BAAI/bge-m3) in the skilled trades domain.
It checks whether a **new quotation item** (from a GAEB bill of quantities, as free text or
from a customer e-mail) finds the matching **old items and articles** by vector search,
so that a quotation can be built from existing items.

## Format

Both files are JSON Lines (UTF-8, LF, no BOM), one object per line.

`corpus.jsonl` – documents:

```json
{"id":"p0001","kind":"position","unit":"m","text":"Kurztext\nLangtext …"}
{"id":"a0001","kind":"article","unit":"m","text":"Bezeichnung\nBeschreibung …"}
```

- `kind`: `position` (work item: short text, often with a long text after a
  line break) or `article` (catalog article: name + short description).
- `unit`: billing unit (m, lfm, m², m³, Stk, St, Stck, psch, h, kg, l).

`queries.jsonl` – queries with relevance judgments:

```json
{"id":"q001","text":"…","rel":{"p0001":2,"p0002":2,"a0001":1}}
```

- `2` = same work or same article (worded differently, possibly a different
  unit, abbreviation or word order).
- `1` = related or partly matching (different dimension or version, e.g.
  DN 20 instead of DN 15, or accessories; a bare article for a "supply and
  install" query).
- not listed = irrelevant. Every query has at least two grade-2 documents (rule
  of this corpus; the generator generally enforces at least one).

## Structure

- Trades: plumbing, heating, ventilation, electrical, tiling, painting, drywall,
  shell construction/demolition, roofing/sheet metal, outdoor works, plus general items
  (hourly labor, site setup).
- About 55 work/article concepts with 2–6 differently worded
  documents each (synonyms, abbreviations such as DN, inkl., einschl., lfm, Stck, psch;
  sometimes short text only, sometimes with long text, sometimes in article style).
- Dimension and version variants (e.g. DN 15/20/25, Typ 22/33,
  60/80 cm, 3 x 1,5 / 5 x 2,5 mm², Q2/Q3/Q4) are rated as grade 1.
- **Hard negatives**: documents with almost the same wording but a different work
  (Kupferrohr verlegen ↔ demontieren, Heizkörper montieren ↔ lackieren ↔
  ab-/anbauen, Wand verputzen ↔ Putz abschlagen, tapezieren ↔ Tapete entfernen,
  Pflaster verlegen ↔ aufnehmen, Dachrinne montieren ↔ reinigen, Schlitze fräsen ↔
  zuputzen u. a.). They are never grade 2 for the respective other work; 11 of
  them belong to no query and serve only as distractors.
- Queries: terse in bill-of-quantities style, as short text + long text (4 queries) or
  colloquial as in a customer e-mail.
- Long texts at most about 100 words (model limit 512 tokens); 46 items
  (about 21 %) have a long text of more than 80 words.

## Counts

| Quantity | Count |
|---|---|
| Documents in total | 262 |
| of which `position` | 221 |
| of which `article` | 41 |
| Queries | 58 |
| Relevance pairs grade 2 | 164 |
| Relevance pairs grade 1 | 104 |

## Provenance

Written in-house, synthetic. No texts from STLB, standards or manufacturer catalogs,
no brand names, no personal or customer data. Standards are named only by their code
(e.g. „nach DIN 1988“). Dimensions, prices and performance values are chosen to be
realistic, but fictitious.

## Reference (`reference/`)

Generated with the oracle sqlite-lembed + bge-m3 q8 via `scripts/embed-reference.sh`
(generator `demos/rag/embed-reference.lpr`); checked in so that later tests work without the
lembed runtime:

- `manifest.json` – reference contract (model SHA-256, runtime, dimension, pooling,
  normalization, metric, token limit, special-token behavior), SHA-256 of these two
  `.jsonl` files, statistics, metrics (nDCG@10, Recall@10, MRR = reciprocal rank of the first grade-2 document) and the exact
  top 20 per query (`[id, score, grade]`).
- `docs.f32`, `queries.f32` – vectors as float32 little-endian, row by row in
  file order.

Whoever changes the corpus, the model or the lembed build regenerates the reference and states
the new metrics in the commit. Details: [docs/native-embeddings.md](../../../docs/native-embeddings.md),
section "Referenzvertrag v1".
