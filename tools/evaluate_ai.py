#!/usr/bin/env python3
"""Evaluate AI shader generation quality using the benchmark suite.

Usage:
    python tools/evaluate_ai.py
    python tools/evaluate_ai.py --provider ollama --model llama3
    python tools/evaluate_ai.py --cases data/benchmark_cases.json --output results.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from luxc.ai.benchmark import load_benchmark_cases, run_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AI shader generation quality"
    )
    parser.add_argument(
        "--cases", type=Path, default=None,
        help="Benchmark cases JSON file (default: data/benchmark_cases.json)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output results to JSON file",
    )
    parser.add_argument(
        "--provider", type=str, default=None,
        help="AI provider to evaluate",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model to evaluate",
    )
    parser.add_argument(
        "--base-url", type=str, default=None,
        help="Base URL for OpenAI-compatible endpoints",
    )
    args = parser.parse_args()

    cases = load_benchmark_cases(args.cases)
    if not cases:
        print("No benchmark cases found.", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(cases)} benchmark cases...")
    print(f"Provider: {args.provider or 'default'}")
    print(f"Model: {args.model or 'default'}")
    print()

    results = run_benchmark(
        cases=cases,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
    )

    # Print summary
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total cases:              {results['total_cases']}")
    print(f"Compilation rate:         {results['compilation_rate']:.1%}")
    print(f"Avg parameter accuracy:   {results['average_parameter_accuracy']:.1%}")
    print(f"Energy conservation rate:  {results['energy_conservation_rate']:.1%}")
    print(f"Layer correctness rate:   {results['layer_correctness_rate']:.1%}")
    print()

    # Per-case results
    print(f"{'Case ID':<25} {'Compiled':>8} {'Accuracy':>8} {'Energy':>8} {'Layers':>8}")
    print("-" * 60)
    for score in results["scores"]:
        compiled = "PASS" if score["compiled"] else "FAIL"
        accuracy = f"{score['accuracy']:.0%}" if score["compiled"] else "-"
        energy = "OK" if score["energy_ok"] else "FAIL"
        layers = "OK" if score["layers_ok"] else "MISS"
        print(f"{score['case_id']:<25} {compiled:>8} {accuracy:>8} {energy:>8} {layers:>8}")

    # Save to file if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
