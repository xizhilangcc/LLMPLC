from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def add_arguments(parser):
    parser.add_argument('--read1', type=Path, required=True, help='Paired FASTQ R1 (.fastq/.fq, optionally .gz).')
    parser.add_argument('--read2', type=Path, required=True, help='Paired FASTQ R2 (.fastq/.fq, optionally .gz).')
    parser.add_argument('--palace-env', type=Path, help='PALACE Conda prefix. Defaults to PALACE on PATH or a sibling palace_env.')
    parser.add_argument('--palace-resources', type=Path, required=True, help='Directory with the PALACE reference database, protein database and GCN checkpoint.')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--min-phage-length', type=int, default=10_000, help='PALACE minimum phage length in bp.')
    parser.add_argument('--spades-memory-gb', type=int, help='Override the PALACE v1.0 SPAdes memory cap, in GB.')
    parser.add_argument('--palace-scratch', type=Path, default=Path('/tmp'), help='Existing scratch directory with space for PALACE assembly.')


def require_nonempty(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'File is missing or empty: {path}')


def assemble(args, work):
    read_paths = [args.read1.resolve(), args.read2.resolve()]
    for path in read_paths:
        require_nonempty(path)
        if not path.name.lower().endswith(('.fastq', '.fq', '.fastq.gz', '.fq.gz')):
            raise ValueError(f'Expected paired FASTQ reads: {path}')
    if read_paths[0].samefile(read_paths[1]):
        raise ValueError('R1 and R2 must be different files')
    if args.threads < 1 or args.min_phage_length < 1 or (
        args.spades_memory_gb is not None and args.spades_memory_gb < 1
    ):
        raise ValueError('Threads, minimum phage length, and SPAdes memory must be positive')
    resources = args.palace_resources.resolve()
    for name in ('phage_refs.fasta', 'phage_refs.fasta.fai', 'GCN_model_retrained.pt'):
        require_nonempty(resources / name)
    protein_db = resources / 'protein_db'
    if not protein_db.is_dir() or not any(p.is_file() and p.stat().st_size for p in protein_db.iterdir()):
        raise ValueError(f'PALACE protein database is missing or empty: {protein_db}')
    if args.palace_env is not None:
        palace = args.palace_env.resolve() / 'bin/palace'
    else:
        on_path = shutil.which('palace')
        sibling_palace = Path(sys.prefix).parent / 'palace_env/bin/palace'
        palace = Path(on_path).absolute() if on_path else sibling_palace
    if not palace.is_file() or not os.access(palace, os.X_OK):
        raise ValueError(f'PALACE executable is missing or not executable: {palace}. '
                         'Install palace_env or set --palace-env to its Conda prefix.')
    prefix = palace.parent.parent
    scratch = args.palace_scratch.resolve()
    if not scratch.is_dir():
        raise ValueError(f'Scratch directory does not exist: {scratch}')
    for path in (prefix, scratch):
        if not re.fullmatch(r'[/A-Za-z0-9_.+-]+', str(path)):
            raise ValueError(f'Use a PALACE environment and scratch path with ASCII letters, digits, /, _, ., + or -: {path}')

    work.mkdir(parents=True, exist_ok=True)
    log = work / 'palace.log'
    stage = Path(tempfile.mkdtemp(prefix='llmplc-palace-', dir=scratch))
    try:
        fastqs = []
        for i, source in enumerate(read_paths, start=1):
            alias = stage / (f'R{i}.fastq.gz' if source.name.lower().endswith('.gz') else f'R{i}.fastq')
            alias.symlink_to(source)
            fastqs.append(alias)
        for name in ('phage_refs.fasta', 'phage_refs.fasta.fai', 'GCN_model_retrained.pt', 'protein_db'):
            (stage / name).symlink_to(resources / name)
        output = stage / 'assembly'
        config = stage / 'palace.config'
        config.write_text(
            f'fastq1={fastqs[0]}\nfastq2={fastqs[1]}\n'
            f'phagedb={stage}/phage_refs.fasta\nprotein_db={stage}/protein_db\n'
            f'gcn_model={stage}/GCN_model_retrained.pt\n'
            f'out_dir={output}\nprefix=phages\nthreads={args.threads}\n'
            f'ENV_PREFIX={prefix}\nMIN_LEN={args.min_phage_length}\n',
            encoding='ascii',
        )
        source = palace.read_text(encoding='utf-8')
        if args.spades_memory_gb is not None:
            setting = '-t "$threads" -m 200'
            if setting not in source:
                raise ValueError('Unrecognized PALACE SPAdes command. Use PALACE v1.0 or omit --spades-memory-gb.')
            source = source.replace(setting, f'-t "$threads" -m {args.spades_memory_gb}', 1)
        env = os.environ.copy()
        env['PATH'] = f'{prefix}/bin{os.pathsep}{env.get("PATH", "")}'
        env['CONDA_PREFIX'] = str(prefix)
        env['CUDA_VISIBLE_DEVICES'] = ''
        env['OMP_NUM_THREADS'] = str(args.threads)
        scorer = prefix / 'share/palace/scripts/phage_scoring.py'
        scoring_code = scorer.read_text(encoding='utf-8')
        override = 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"'
        if override in scoring_code:
            patched_scorer = stage / scorer.name
            patched_scorer.write_text(
                scoring_code.replace(override, 'os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")', 1),
                encoding='utf-8',
            )
            if '$SCRIPTS/phage_scoring.py' not in source:
                raise ValueError('Unrecognized PALACE scoring command. Install the documented PALACE v1.0 package.')
            source = source.replace('$SCRIPTS/phage_scoring.py', str(patched_scorer), 1)
            env['PYTHONPATH'] = str(scorer.parent)
        launcher = stage / 'palace.sh'
        launcher.write_text(source, encoding='utf-8')
        print(f'[PALACE] Running assembly. Log: {log}', flush=True)
        with log.open('w', encoding='utf-8') as handle:
            subprocess.run(['bash', str(launcher), '--config', str(config)], cwd=stage, env=env,
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
        final = output / 'final_result/phages_final.fasta'
        if not final.is_file():
            raise RuntimeError('PALACE exited without producing final_result/phages_final.fasta')
        fasta = work / 'phages.fasta'
        shutil.copyfile(final, fasta)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f'{exc}. PALACE log: {log}. Scratch retained: {stage}') from exc
    shutil.rmtree(stage)
    return fasta
