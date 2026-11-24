#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ProteinDataset(Dataset):
    def __init__(self, fasta: Path):
        self.items = [(rec.id, str(rec.seq)) for rec in SeqIO.parse(str(fasta), 'fasta')]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description='Embed Fragment_proteins.faa using local ESM2.')
    p.add_argument('--work-dir', type=Path, default=Path('LLMPhage'))
    p.add_argument('--protein-fasta', type=Path, default=None)
    p.add_argument('--fragment-fasta', type=Path, default=None)
    p.add_argument('--contig-sentence', type=Path, default=None)
    p.add_argument('--out-h5', type=Path, default=None)
    p.add_argument('--model-dir', type=Path, default=script_dir.parent / 'weights/esm2_t33_650M_UR50D')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--num-workers', type=int, default=1)
    p.add_argument('--max-length', type=int, default=1024)
    p.add_argument('--remove-intermediate', action='store_true', help='Remove per-protein and per-fragment .pt/.pkl after H5 is written.')
    return p.parse_args()


def load_fragment_protein_map(path: Path) -> dict[str, list[str]]:
    out = {}
    with path.open(newline='', encoding='utf-8') as fh:
        for row in csv.reader(fh):
            if row:
                out[row[0]] = [x for x in row[1:] if x]
    return out


def tensor_from_saved(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, torch.Tensor):
                return v
    raise TypeError(f'Cannot parse tensor from {path}')


def mean_pool_tensor(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 1:
        return t.to(torch.float32).numpy()
    if t.dim() == 2:
        return t.mean(dim=0).to(torch.float32).numpy()
    if t.dim() >= 3:
        return t.reshape(t.shape[0], t.shape[1], -1).mean(dim=1).squeeze(0).to(torch.float32).numpy()
    raise ValueError(f'Unexpected tensor shape: {tuple(t.shape)}')


def embed_proteins(args, protein_dir: Path) -> None:
    from transformers import AutoTokenizer, EsmModel

    dataset = ProteinDataset(args.protein_fasta)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = EsmModel.from_pretrained(str(args.model_dir), add_pooling_layer=False).to(device)
    model.eval()
    protein_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for names, seqs in tqdm(loader, desc='ESM2 protein embedding'):
            inputs = tokenizer(list(seqs), return_tensors='pt', padding='max_length', max_length=args.max_length, truncation=True).to(device)
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state.cpu()
            lengths = inputs['attention_mask'].sum(dim=1).cpu()
            mask = inputs['attention_mask'].cpu()
            mask[:, 0] = 0
            for i, length in enumerate(lengths):
                if length > 1:
                    mask[i, length - 1] = 0
            pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / (lengths - 2).clamp(min=1).unsqueeze(-1)
            for name, vec in zip(names, pooled):
                torch.save({'data': vec.unsqueeze(0)}, protein_dir / f'{name}_embedding.pkl')


def build_fragment_h5(args, protein_dir: Path, fragment_dir: Path) -> tuple[int, int]:
    frag_to_proteins = load_fragment_protein_map(args.contig_sentence)
    fragment_dir.mkdir(parents=True, exist_ok=True)
    ids, feats = [], []
    missing = 0
    for rec in tqdm(list(SeqIO.parse(str(args.fragment_fasta), 'fasta')), desc='Build fragment ESM2 H5'):
        proteins = frag_to_proteins.get(rec.id, [])
        tensors = []
        for protein in proteins:
            path = protein_dir / f'{protein}_embedding.pkl'
            if path.exists():
                tensors.append(tensor_from_saved(path).squeeze(0))
        if not tensors:
            missing += 1
            continue
        seq_tensor = torch.stack(tensors, dim=0)
        torch.save({'data': seq_tensor}, fragment_dir / f'{rec.id}_embedding.pt')
        ids.append(rec.id)
        feats.append(mean_pool_tensor(seq_tensor))

    X = np.stack(feats, axis=0).astype(np.float32) if feats else np.empty((0, 1280), dtype=np.float32)
    args.out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out_h5, 'w') as f:
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('ids', data=np.array(ids, dtype=object), dtype=dt)
        f.create_dataset('features', data=X, compression='gzip')
    return len(ids), missing


def main() -> int:
    args = parse_args()
    args.protein_fasta = args.protein_fasta or args.work_dir / 'Fragment_proteins.faa'
    args.fragment_fasta = args.fragment_fasta or args.work_dir / 'Fragment_sequences.fasta'
    args.contig_sentence = args.contig_sentence or args.work_dir / 'contig_sentence.csv'
    args.out_h5 = args.out_h5 or args.work_dir / 'esm33_embedding.h5'
    protein_dir = args.work_dir / 'esm33_per_residual_embedding'
    fragment_dir = args.work_dir / 'sequence_embedding_esm33'

    has_proteins = next(SeqIO.parse(str(args.protein_fasta), 'fasta'), None) is not None
    if has_proteins:
        embed_proteins(args, protein_dir)
    else:
        args.out_h5.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.out_h5, 'w') as f:
            f.create_dataset('ids', data=np.array([], dtype=object), dtype=h5py.string_dtype('utf-8'))
            f.create_dataset('features', shape=(0, 1280), dtype='float32')
        print(f'[OK] no proteins; Evo2-only route -> {args.out_h5}')
        return 0
    n, missing = build_fragment_h5(args, protein_dir, fragment_dir)
    print(f'[OK] {args.out_h5}: fragment_with_esm={n}, fragment_without_protein={missing}')
    if args.remove_intermediate:
        shutil.rmtree(protein_dir, ignore_errors=True)
        shutil.rmtree(fragment_dir, ignore_errors=True)
        print('[OK] removed ESM2 intermediate directories')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
