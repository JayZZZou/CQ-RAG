# Public datasets

CQ-RAG uses HotpotQA and MuSiQue, which are public datasets that could download from their official sources under the applicable dataset terms.

- HotpotQA: save the selected labeled split as `hotpotqa.json`, retaining `_id`, `question`, `answer`, `context` and `supporting_facts`.
- MuSiQue: save the selected labeled, answerable split as `musique.json`, retaining `id`, `question`, `answer`, optional `answer_aliases`, and `paragraphs` with `idx`, `title`, `paragraph_text` and `is_supporting`; records marked `answerable: false` are not supported.

Each file must contain a JSON array of records or an object with a top-level `data` array; if the official download uses JSON Lines, convert the entire selected split to a JSON array without sampling or truncation.
Place the files in this directory or supply a compatible file with `--dataset-path`; ALL records in the supplied file are processed in their original order.

