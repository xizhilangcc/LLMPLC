#!/usr/bin/env python3
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    p = argparse.ArgumentParser(description='Download the two official checkpoints needed by LLMPLC.')
    p.add_argument('--weights-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'weights')
    args = p.parse_args()
    for repo, folder, files in [
        ('arcinstitute/evo2_7b_base', 'evo2_7b_base', ['evo2_7b_base.pt', 'config.json', 'README.md']),
        ('facebook/esm2_t33_650M_UR50D', 'esm2_t33_650M_UR50D',
         ['model.safetensors', 'config.json', 'special_tokens_map.json', 'tokenizer_config.json', 'vocab.txt', 'README.md']),
    ]:
        snapshot_download(repo_id=repo, local_dir=str(args.weights_dir / folder), allow_patterns=files)


if __name__ == '__main__':
    main()
