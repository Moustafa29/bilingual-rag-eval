"""Which question file each retrieval corpus is evaluated with."""

QUESTION_FILES = {
    "xquad": "xquad.jsonl",
    "unpc": "unpc_pilot.jsonl",
    # The same questions over the Arabic text as distributed, before digit-group correction.
    "unpc_uncorrected": "unpc_pilot.jsonl",
}
