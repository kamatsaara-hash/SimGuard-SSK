"""Validate IDs, references, and counts in SIMGuard graph CSV outputs.

Run from the SIMGuard project root:
    python scripts/audit_graphs.py --limit 1
    python scripts/audit_graphs.py

An event's event_id MUST exactly equal its node_id in nodes.csv.  This
script also verifies that every edge endpoint and event reference exists.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def check_one(summary_path):
    folder = summary_path.parent
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    nodes = {}
    node_counts = Counter()
    for row in read_rows(folder / 'nodes.csv'):
        node_id = row['node_id']
        if node_id in nodes:
            raise ValueError(f'{folder}: duplicate node ID: {node_id}')
        nodes[node_id] = row['node_type']
        node_counts[row['node_type']] += 1
    if len(nodes) != summary['nodes'] or dict(node_counts) != summary['node_types']:
        raise ValueError(f'{folder}: node totals/types do not match summary')

    attrs = 0
    attr_ids = set()
    for row in read_rows(folder / 'event_attributes.csv'):
        event_id = row['event_id']
        if nodes.get(event_id) != 'event':
            raise ValueError(f'{folder}: event attribute refers to a missing event node: {event_id}')
        if event_id in attr_ids:
            raise ValueError(f'{folder}: repeated event attribute: {event_id}')
        attr_ids.add(event_id)
        attrs += 1
    if attrs != summary['events'] or attrs != node_counts['event']:
        raise ValueError(f'{folder}: event attribute count disagrees with nodes/summary')

    relations = Counter()
    for row in read_rows(folder / 'edges.csv'):
        src, dst, event_id = row['source'], row['target'], row['event_id']
        if src not in nodes or dst not in nodes:
            raise ValueError(f'{folder}: missing edge endpoint: {src} -> {dst}')
        if event_id not in attr_ids:
            raise ValueError(f'{folder}: edge references missing event attribute: {event_id}')
        relation = row['relation']
        if relation == 'originates_event' and dst != event_id:
            raise ValueError(f'{folder}: originates_event target != event ID')
        if relation == 'event_targets' and src != event_id:
            raise ValueError(f'{folder}: event_targets source != event ID')
        relations[relation] += 1
    if sum(relations.values()) != summary['edges'] or dict(relations) != summary['edge_types']:
        raise ValueError(f'{folder}: edge totals/types do not match summary')

    targets, fraud = 0, 0
    for row in read_rows(folder / 'targets.csv'):
        if nodes.get(row['node_id']) != 'subscriber':
            raise ValueError(f'{folder}: target is not a subscriber node')
        if row['label'] not in ('0', '1') or row['split'] not in ('train', 'val', 'test'):
            raise ValueError(f'{folder}: invalid target label or split')
        targets += 1
        fraud += int(row['label'])
    if targets != summary['targets'] or fraud != summary['fraud_targets']:
        raise ValueError(f'{folder}: target counts do not match summary')
    print(f"PASS {summary['scenario_id']}: {attrs:,} event nodes, "
          f"{sum(relations.values()):,} edges, {targets:,} targets ({fraud:,} fraud)")
    return summary


def main():
    p = argparse.ArgumentParser(description='Audit SIMGuard generated graph files')
    p.add_argument('--graphs', type=Path, default=Path('data/graphs'))
    p.add_argument('--limit', type=int, default=None)
    args = p.parse_args()
    paths = sorted(args.graphs.rglob('summary.json'))
    if not paths:
        p.error(f'No summary.json files found under {args.graphs}')
    if args.limit is not None:
        if args.limit < 1: p.error('--limit must be >=1')
        paths = paths[:args.limit]
    for path in paths:
        check_one(path)
    print(f'All {len(paths)} audited graphs PASSED.')


if __name__ == '__main__':
    main()
