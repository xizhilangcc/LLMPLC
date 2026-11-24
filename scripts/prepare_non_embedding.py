#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
from collections import defaultdict
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Prepare sequence fragments and Prodigal proteins.')
    p.add_argument('--input-fasta', required=True, type=Path, help='Input genome/contig FASTA file.')
    p.add_argument('--work-dir', type=Path, default=Path('LLMPhage'), help='Output working directory.')
    p.add_argument('--fragment-length', type=int, default=1500, help='Fixed fragment length. Short tail fragments are discarded.')
    p.add_argument('--skip-prodigal', action='store_true', help='Only create fragment FASTA and mapping files.')
    return p.parse_args()


def split_fragments(input_fasta: Path, fragment_fasta: Path, all_map_csv: Path, fragment_length: int) -> int:
    fragment_fasta.parent.mkdir(parents=True, exist_ok=True)
    if fragment_length <= 0:
        raise ValueError('fragment-length must be positive')
    seen = set()
    for rec in SeqIO.parse(str(input_fasta), 'fasta'):
        if rec.id in seen or '/' in rec.id or chr(92) in rec.id:
            raise ValueError(f'Duplicate or unsupported FASTA ID: {rec.id}')
        seen.add(rec.id)
    if not seen:
        raise ValueError('Input FASTA has no records')
    n = 0
    with fragment_fasta.open('w', encoding='utf-8') as out_fa, all_map_csv.open('w', newline='', encoding='utf-8') as out_csv:
        writer = csv.writer(out_csv)
        for rec in SeqIO.parse(str(input_fasta), 'fasta'):
            seq = str(rec.seq).upper()
            frag_ids = []
            for start in range(0, len(seq), fragment_length):
                frag = seq[start:start + fragment_length]
                if len(frag) != fragment_length:
                    continue
                frag_id = f'{rec.id}_{start // fragment_length + 1}'
                SeqIO.write(SeqRecord(Seq(frag), id=frag_id, description=''), out_fa, 'fasta')
                frag_ids.append(frag_id)
                n += 1
            writer.writerow([rec.id, *frag_ids])
    return n


def run_prodigal(fragment_fasta: Path, protein_faa: Path) -> None:
    cmd = ['prodigal', '-i', str(fragment_fasta), '-a', str(protein_faa), '-p', 'meta', '-q', '-o', os.devnull]
    subprocess.check_call(cmd)


def build_fragment_protein_sentence(protein_faa: Path, out_csv: Path) -> int:
    frag_to_proteins: dict[str, list[str]] = defaultdict(list)
    for rec in SeqIO.parse(str(protein_faa), 'fasta'):
        protein_id = rec.id
        parts = protein_id.split('_')
        if len(parts) < 2:
            continue
        fragment_id = '_'.join(parts[:-1])
        frag_to_proteins[fragment_id].append(protein_id)

    with out_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        for fragment_id, proteins in sorted(frag_to_proteins.items()):
            writer.writerow([fragment_id, *proteins])
    return len(frag_to_proteins)


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    fragment_fasta = args.work_dir / 'Fragment_sequences.fasta'
    protein_faa = args.work_dir / 'Fragment_proteins.faa'
    contig_sentence = args.work_dir / 'contig_sentence.csv'
    all_contig_sentence = args.work_dir / 'All_contig_sentence.csv'

    n_frag = split_fragments(args.input_fasta, fragment_fasta, all_contig_sentence, args.fragment_length)
    print(f'[OK] fragments: {n_frag} -> {fragment_fasta}')
    print(f'[OK] original-contig map -> {all_contig_sentence}')

    if not args.skip_prodigal:
        if n_frag:
            run_prodigal(fragment_fasta, protein_faa)
        else:
            protein_faa.write_text('', encoding='utf-8')
        n_with_protein = build_fragment_protein_sentence(protein_faa, contig_sentence)
        print(f'[OK] proteins -> {protein_faa}')
        print(f'[OK] fragment-protein map: {n_with_protein} fragments -> {contig_sentence}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
