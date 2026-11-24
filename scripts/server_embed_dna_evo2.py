#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description='Embed Fragment_sequences.fasta using Evo2.')
    p.add_argument('--fasta', type=Path, default=Path('LLMPhage/Fragment_sequences.fasta'))
    p.add_argument('--out-h5', type=Path, default=Path('LLMPhage/evo2_embedding.h5'))
    p.add_argument('--model-path', type=Path, default=script_dir.parent / 'weights/evo2_7b_base/evo2_7b_base.pt')
    p.add_argument('--model-name', default='evo2_7b_base')
    p.add_argument('--layer-name', default='blocks.28.mlp.l3')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--exclude-intervals', type=Path, default=None,
                   help='TSV with sequence_id and removed_intervals (0-based half-open spans).')
    return p.parse_args()


def load_excluded_intervals(path: Path | None) -> dict[str, list[tuple[int, int]]]:
    if path is None:
        return {}
    excluded = {}
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            spans = []
            for item in (row.get('removed_intervals') or '').split(';'):
                if not item:
                    continue
                start, end = item.split('-', 1)
                spans.append((int(start), int(end)))
            excluded[row['sequence_id']] = spans
    return excluded


def process_sequence(sequence: str, model, layer_name: str, device: str,
                     excluded: list[tuple[int, int]] | None = None) -> np.ndarray | None:
    input_ids = torch.tensor(model.tokenizer.tokenize(sequence), dtype=torch.int).unsqueeze(0).to(device)
    with torch.no_grad():
        _, embeddings = model(input_ids, return_embeddings=True, layer_names=[layer_name])
    emb = embeddings[layer_name].squeeze(0).to(torch.float32).cpu().numpy()
    excluded = excluded or []
    if emb.shape[0] != len(sequence):
        raise ValueError(
            f'Evo2 returned {emb.shape[0]} positions for {len(sequence)} bases; '
            'base-level interval masking requires one embedding per nucleotide.'
        )
    keep = np.ones(len(sequence), dtype=bool)
    for start, end in excluded:
        if start < 0 or end < start or end > len(sequence):
            raise ValueError(f'Invalid excluded interval {start}-{end} for sequence length {len(sequence)}')
        keep[start:end] = False
    if not keep.any():
        return None
    return np.mean(emb[keep], axis=0).astype(np.float32)


def main() -> int:
    args = parse_args()
    for key in ['fasta', 'out_h5', 'model_path', 'exclude_intervals']:
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    records = list(SeqIO.parse(str(args.fasta), 'fasta'))
    args.out_h5.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        with h5py.File(args.out_h5, 'w') as f:
            f.create_dataset('ids', data=np.array([], dtype=object), dtype=h5py.string_dtype('utf-8'))
            f.create_dataset('features', shape=(0, 4096), dtype='float32')
        print(f'[OK] empty DNA input -> {args.out_h5}')
        return 0
    if not args.model_path.is_file():
        raise FileNotFoundError(f'{args.model_path}; run scripts/download_weights.py first')
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Evo2 embedding requires CUDA; CPU supports cached-feature prediction only')
    torch.cuda.set_device(torch.device(args.device))
    from evo2 import Evo2
    import evo2

    os.chdir(Path(evo2.__file__).resolve().parent)

    args.out_h5.parent.mkdir(parents=True, exist_ok=True)
    model = Evo2(model_name=args.model_name, local_path=str(args.model_path))
    excluded = load_excluded_intervals(args.exclude_intervals)
    ids, features = [], []
    model.model.eval()
    for rec in tqdm(records, desc='Evo2 DNA embedding'):
        feature = process_sequence(
            str(rec.seq), model, args.layer_name, args.device,
            excluded.get(rec.id, []),
        )
        if feature is None:
            print(f'[SKIP] {rec.id}: BACPHLIP intervals cover the entire DNA sequence', flush=True)
            continue
        ids.append(rec.id)
        features.append(feature)

    if not features:
        raise RuntimeError('No DNA sequences remain after excluding full-length BACPHLIP hits.')
    X = np.stack(features, axis=0).astype(np.float32)
    with h5py.File(args.out_h5, 'w') as f:
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('ids', data=np.array(ids, dtype=object), dtype=dt)
        f.create_dataset('features', data=X, compression='lzf')
    print(f'[OK] {args.out_h5}: N={len(ids)}, D={X.shape[1]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
