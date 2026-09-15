"""Check the environment before any retrieval run, and fail immediately with a clear message.

    python scripts/colab_check.py                                 # laptop: GPU optional
    python scripts/colab_check.py --require-cuda                  # Colab: a missing GPU is an error
    python scripts/colab_check.py --require-cuda --verify-data    # Colab, after rebuilding the corpus

--verify-data compares the frozen question file and both rebuilt chunk files with the fingerprints in
configs/pilot.yaml. A mismatch means the questions' gold chunk ids would point at different text; for
example, a different tokenizers version changed chunk boundaries. Retrieval must not run then.

Everything is imported here, up front, so a broken install fails before a multi-gigabyte model download
rather than deep inside the encoder. The most likely failure on Colab is torchvision orphaned by a torch
upgrade: it shows up as an import error mentioning torchvision (often "operator torchvision::nms does not
exist") when transformers is imported. The fix is the constrained install in docs/colab.md, not a
re-run.

Nothing here downloads a model: importing `sentence_transformers` does not load one.
"""

from __future__ import annotations

import argparse
import importlib
import sys

REQUIRED = ["torch", "transformers", "tokenizers", "sentence_transformers", "faiss", "snowballstemmer", "scipy", "numpy", "yaml", "httpx", "rageval"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--verify-data", action="store_true")
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()

    problems, versions = [], {}
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "installed")
        except Exception as error:  # an import can fail with more than ImportError (e.g. RuntimeError from a torch mismatch)
            hint = ""
            if "torchvision" in str(error):
                hint = " -> torchvision does not match torch; reinstall with the constrained install in docs/colab.md"
            elif name == "rageval":
                hint = " -> run `pip install -e . --no-deps` from the repository root"
            problems.append(f"{name}: {type(error).__name__}: {error}{hint}")

    try:
        import torchvision  # noqa: F401  (only present on Colab-like environments; a mismatch must surface now)

        versions["torchvision"] = torchvision.__version__
    except ImportError as error:
        if "torchvision" in str(error) and "No module named" not in str(error):
            problems.append(f"torchvision: {error} -> torchvision does not match torch; use the constrained install in docs/colab.md")
    except Exception as error:
        problems.append(f"torchvision: {type(error).__name__}: {error} -> torchvision does not match torch; use the constrained install in docs/colab.md")

    if "sentence_transformers" in versions:
        try:
            from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: F401
        except Exception as error:
            problems.append(f"sentence_transformers classes: {type(error).__name__}: {error}")

    if "torch" in versions:
        import torch

        cuda = torch.cuda.is_available()
        versions["cuda"] = torch.cuda.get_device_name(0) if cuda else "not available"
        if args.require_cuda and not cuda:
            problems.append("CUDA: no GPU visible to torch -> Runtime > Change runtime type > T4 GPU, then restart")

    if args.verify_data and "rageval" in versions:
        from pathlib import Path

        from rageval.io import jsonl_fingerprint, load_config

        expected = load_config(args.config).get("fingerprints") or {}
        if not expected:
            problems.append(f"no fingerprints in {args.config}")
        for path, fingerprint in expected.items():
            if not Path(path).exists():
                hint = "upload it from the laptop" if "questions" in path else "run scripts/build_unpc.py"
                problems.append(f"{path}: missing -> {hint}")
            elif jsonl_fingerprint(path) != fingerprint:
                problems.append(
                    f"{path}: fingerprint differs from the frozen data -> do not run retrieval; the questions' gold "
                    "chunk ids would point at different text (check tokenizers==0.23.2, or re-upload the question file)"
                )
            else:
                versions[f"data {path}"] = "matches"

    print("versions:", ", ".join(f"{k} {v}" for k, v in versions.items()))
    if problems:
        print("\nSETUP FAILED; fix these before running anything:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        sys.exit(1)
    print("setup OK")


if __name__ == "__main__":
    main()
