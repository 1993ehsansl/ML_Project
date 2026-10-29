"""Command line entry point."""
import argparse
import json
from .experiments import run_all, run_regression, run_classification, run_clustering


def main():
    parser = argparse.ArgumentParser(description='ML benchmark suite')
    parser.add_argument('--task', choices=['all', 'regression', 'classification', 'clustering'], default='all')
    parser.add_argument('--out', default='artifacts')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick', action='store_true', help='Smoke test with fewer epochs')
    parser.add_argument('--track', action='store_true', help='Log full suite to MLflow')
    args = parser.parse_args()
    if args.track and args.task != 'all':
        parser.error('--track currently requires --task all')
    functions = {'all': run_all, 'regression': run_regression,
                 'classification': run_classification, 'clustering': run_clustering}
    kwargs = {'out': args.out, 'seed': args.seed, 'quick': args.quick}
    if args.task == 'all':
        kwargs['track'] = args.track
    result = functions[args.task](**kwargs)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
