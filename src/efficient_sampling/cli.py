from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


BENCHMARK_SHARED_FILES = (
    "main_workflow.py",
    "prep.py",
    "dataset.py",
    "interpolator.py",
    "plotter.py",
    "plot_func.py",
)


def discover_repo_root(explicit: str | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()

    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parents[2],
    ]
    for base in candidates:
        for candidate in (base, *base.parents):
            if (candidate / "pyproject.toml").exists() and (candidate / "code").is_dir():
                return candidate
    raise SystemExit("Could not locate the repository root. Pass --repo-root explicitly.")


def benchmark_case_dir(repo_root: Path, table: str, benchmark: str, sampler: str) -> Path:
    case_dir = repo_root / "code" / table / benchmark / sampler
    if not case_dir.is_dir():
        raise SystemExit(f"Unknown benchmark case: {table}/{benchmark}/{sampler}")
    return case_dir


def benchmark_output_dir(repo_root: Path, benchmark: str, sampler: str, results_dir: str) -> Path:
    return repo_root / results_dir / benchmark / sampler


def stage_benchmark(repo_root: Path, table: str, benchmark: str, sampler: str, output: Path) -> Path:
    case_dir = benchmark_case_dir(repo_root, table, benchmark, sampler)
    shared_dir = repo_root / "code" / "common" / "benchmark"
    output.mkdir(parents=True, exist_ok=True)

    for name in BENCHMARK_SHARED_FILES:
        shutil.copy2(shared_dir / name, output / name)

    for path in case_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, output / path.name)

    return output


def stage_eos(repo_root: Path, model: str, output: Path) -> Path:
    source = repo_root / "code" / "eos" / model
    if not source.is_dir():
        raise SystemExit(f"Unknown EOS model: {model}")
    shutil.copytree(source, output, dirs_exist_ok=True)
    return output


def stage_md(repo_root: Path, model: str, output: Path) -> Path:
    source = repo_root / "code" / "md" / model
    if not source.is_dir():
        raise SystemExit(f"Unknown MD model: {model}")
    shutil.copytree(source, output, dirs_exist_ok=True)
    return output


def run_python_script(script: str, cwd: Path) -> int:
    result = subprocess.run([sys.executable, script], cwd=str(cwd), check=False)
    return result.returncode


def cmd_list(args: argparse.Namespace) -> int:
    repo_root = discover_repo_root(args.repo_root)
    code_root = repo_root / "code"

    print("Benchmark cases:")
    for table_dir in sorted((code_root / "tableI", code_root / "tableII")):
        for benchmark_dir in sorted(p for p in table_dir.iterdir() if p.is_dir()):
            variants = sorted(p.name for p in benchmark_dir.iterdir() if p.is_dir())
            print(f"  {table_dir.name}/{benchmark_dir.name}: {', '.join(variants)}")

    print("EOS cases:")
    for case in sorted(p.name for p in (code_root / "eos").iterdir() if p.is_dir()):
        print(f"  {case}")

    print("MD cases:")
    for case in sorted(p.name for p in (code_root / "md").iterdir() if p.is_dir()):
        print(f"  {case}")

    return 0


def cmd_stage(args: argparse.Namespace) -> int:
    repo_root = discover_repo_root(args.repo_root)
    output = Path(args.output).expanduser().resolve()

    if args.family == "benchmark":
        stage_benchmark(repo_root, args.table, args.case, args.variant, output)
    elif args.family == "eos":
        stage_eos(repo_root, args.case, output)
    else:
        stage_md(repo_root, args.case, output)

    print(output)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    repo_root = discover_repo_root(args.repo_root)
    results_dir = args.results_dir

    if args.family == "benchmark":
        output = benchmark_output_dir(repo_root, args.case, args.variant, results_dir)
        stage_benchmark(repo_root, args.table, args.case, args.variant, output)
    elif args.family == "eos":
        output = repo_root / results_dir / args.case
        stage_eos(repo_root, args.case, output)
    else:
        output = repo_root / results_dir / args.case
        stage_md(repo_root, args.case, output)

    return run_python_script("main_workflow.py", output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="efficient-sampling")
    parser.add_argument("--repo-root", default=None, help="Path to the repository root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available workflows")
    list_parser.set_defaults(func=cmd_list)

    stage_parser = subparsers.add_parser("stage", help="Stage a workflow into a runnable directory")
    stage_subparsers = stage_parser.add_subparsers(dest="family", required=True)

    stage_benchmark_parser = stage_subparsers.add_parser("benchmark", help="Stage a benchmark workflow")
    stage_benchmark_parser.add_argument("table", choices=("tableI", "tableII"))
    stage_benchmark_parser.add_argument("case")
    stage_benchmark_parser.add_argument("variant")
    stage_benchmark_parser.add_argument("--output", required=True)
    stage_benchmark_parser.set_defaults(func=cmd_stage)

    stage_eos_parser = stage_subparsers.add_parser("eos", help="Stage an EOS workflow")
    stage_eos_parser.add_argument("case")
    stage_eos_parser.add_argument("--output", required=True)
    stage_eos_parser.set_defaults(func=cmd_stage)

    stage_md_parser = stage_subparsers.add_parser("md", help="Stage an MD workflow")
    stage_md_parser.add_argument("case")
    stage_md_parser.add_argument("--output", required=True)
    stage_md_parser.set_defaults(func=cmd_stage)

    run_parser = subparsers.add_parser("run", help="Stage and run a workflow")
    run_parser.add_argument("--results-dir", default="results", help="Relative results directory under the repo root")
    run_subparsers = run_parser.add_subparsers(dest="family", required=True)

    run_benchmark_parser = run_subparsers.add_parser("benchmark", help="Run a benchmark workflow")
    run_benchmark_parser.add_argument("table", choices=("tableI", "tableII"))
    run_benchmark_parser.add_argument("case")
    run_benchmark_parser.add_argument("variant")
    run_benchmark_parser.set_defaults(func=cmd_run)

    run_eos_parser = run_subparsers.add_parser("eos", help="Run an EOS workflow")
    run_eos_parser.add_argument("case")
    run_eos_parser.set_defaults(func=cmd_run)

    run_md_parser = run_subparsers.add_parser("md", help="Run an MD workflow")
    run_md_parser.add_argument("case")
    run_md_parser.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
