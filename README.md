# LLMPLC

## Installation

### Step 1: set up the LLMPLC environment

```bash
conda env create -f environment/environment_LLMPLC.yml
conda activate LLMPLC
export CUDA_HOME="$CONDA_PREFIX"
python -m pip install packaging setuptools wheel psutil
MAX_JOBS=4 python -m pip install flash-attn==2.8.0.post2 --no-build-isolation
python -m pip install -r environment/requirements_LLMPLC.txt
```

Alternatively, create the same environment with Mamba, then activate it and run the pip installation commands above:

```bash
mamba env create -f environment/environment_LLMPLC.yml
conda activate LLMPLC
```

The environment includes:

- Python 3.11
- PyTorch 2.7.1 with CUDA 12.9
- Evo2 0.3.0
- Transformers 4.48.3
- XGBoost 2.1.4
- Prodigal 2.6.3

The remaining Python dependencies are listed in `environment/requirements_LLMPLC.txt`.

### Step 2: download pretrained models

Download the pretrained language-model checkpoints before running either workflow from sequence data:

```bash
python scripts/download_weights.py
```

- [Evo2 7B base](https://huggingface.co/arcinstitute/evo2_7b_base), DNA sequence embeddings.
- [ESM2 650M](https://huggingface.co/facebook/esm2_t33_650M_UR50D), protein sequence embeddings.

The download script creates `weights/` and saves the required checkpoints and configuration files there. Approximately 15 GiB of storage is required.

The final lifestyle classifiers are included in the repository:

- `models/merged_best_model.ubj`, the classifier for combined ESM2 and Evo2 features.
- `models/evo2_best_model.ubj`, the fallback classifier for fragments without protein features.
- `models/training_config.json`, model parameters, feature order and label definitions.

### PALACE (required for the metagenomic pipeline)

Install PALACE v1.0 in a separate environment and download its reference databases and GCN model following the [PALACE GitHub instructions](https://github.com/deepomicslab/PALACE).

## Using LLMPLC

### 1. Predict lifestyle from phage contigs

Provide a nucleotide FASTA file containing phage contigs or genomes:

```bash
conda activate LLMPLC
python llmplc.py run \
  --input-fasta phage_contigs.fasta \
  --work-dir runs/phage_contigs \
  --device cuda:0
```

- `--input-fasta`, input phage nucleotide FASTA. Sequence IDs must be unique.
- `--work-dir`, a new or empty output directory.
- `--device`, GPU used for language-model embeddings. Default: `cuda:0`.
- `--threshold`, binary decision threshold for the virulent score. Default: `0.5`.
- `--confidence-threshold`, the selected-class score must exceed this value to retain the lifestyle call. Default: `0.85`.

### 2. Recover phages from metagenomic reads and predict lifestyle

Provide paired-end FASTQ reads and the PALACE resources directory:

```bash
conda activate LLMPLC
python llmplc.py metagenome \
  --read1 sample_R1.fastq.gz \
  --read2 sample_R2.fastq.gz \
  --palace-resources /path/to/PALACE_resources \
  --work-dir runs/sample \
  --threads 8 \
  --spades-memory-gb 48 \
  --device cuda:0
```

- `--read1`, read 1 FASTQ file, optionally gzip-compressed.
- `--read2`, matching read 2 FASTQ file, optionally gzip-compressed.
- `--palace-resources`, directory containing the PALACE reference databases and GCN model.
- `--palace-env`, optional path to the PALACE Conda environment, such as `/path/to/conda/envs/palace_env`.
- `--threads`, threads used by PALACE. Default: `8`.
- `--spades-memory-gb`, optional SPAdes memory limit in GB. The example uses `48`.
- `--min-phage-length`, minimum length of phage sequences recovered by PALACE. Default: `10000` bp.
- `--palace-scratch`, existing temporary directory for PALACE intermediate files. Default: `/tmp`.

## Copyright

Copyright © 2026 LLMPLC contributors.

The project code in this repository is licensed under the MIT License; see [LICENSE](LICENSE).
