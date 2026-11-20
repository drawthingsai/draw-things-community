#!/usr/bin/env python3
"""Write Inkling 2 Small's tokenizer as TiktokenTokenizer resources from its tokenizer.json:
Libraries/BinaryResources/Resources/vocab_inkling2.json (byte-level token -> id, without the added tokens, which
Inkling2SmallTextGenerator lists as special tokens) and merges_inkling2.txt ("#version: 0.2", then one "a b" merge
per line in rank order).

Usage: inkling2_small_tokenizer_resources.py TOKENIZER_JSON OUTPUT_DIR
"""

import json
import sys
from pathlib import Path

tokenizer = json.loads(Path(sys.argv[1]).read_text())
output = Path(sys.argv[2])
model = tokenizer["model"]
assert model["type"] == "BPE" and model["ignore_merges"] and not model.get("byte_fallback")
vocabulary = model["vocab"]
added = {token["content"]: token["id"] for token in tokenizer["added_tokens"]}
assert not set(vocabulary) & set(added), "added tokens are in the vocabulary"
merges = [merge if isinstance(merge, list) else merge.split(" ") for merge in model["merges"]]
# Byte-level tokens spell a space as U+0120, so a space can only separate the two sides.
assert all(len(merge) == 2 and " " not in merge[0] + merge[1] for merge in merges)
(output / "vocab_inkling2.json").write_text(
    json.dumps(vocabulary, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
(output / "merges_inkling2.txt").write_text(
    "#version: 0.2\n" + "\n".join(f"{a} {b}" for a, b in merges) + "\n", encoding="utf-8")
print(f"{len(vocabulary)} tokens, {len(merges)} merges, {len(added)} added tokens "
      f"({min(added.values())}..{max(added.values())})")
print(json.dumps(added, ensure_ascii=False))
