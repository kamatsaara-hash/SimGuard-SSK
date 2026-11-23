"""Build per-scenario, timestamped heterogeneous graphs from FraudZen CDRs.

Run from SIMGuard root:
  python scripts/03_build_graph.py --limit 1
  python scripts/03_build_graph.py

Output per scenario: nodes.csv, edges.csv, event_attributes.csv, targets.csv,
summary.json. Each scenario remains an independent graph. No fraud labels or split
assignments are placed in node/edge features; they live only in targets.csv.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

NODE_FIELDS = ['node_id', 'node_type', 'raw_id']
EDGE_FIELDS = ['source', 'target', 'relation', 'event_id', 'timestamp']
EVENT_FIELDS = ['event_id', 'timestamp', 'CDR_type', 'duration', 'size']
TARGET_FIELDS = ['node_id', 'subscriber_id', 'label', 'split']
REQUIRED = {'scenario_id','timestamp','CDR_type','src_number','dst_number',
            'src_imei','dst_imei','initial_src_cellId','current_src_cellId',
            'initial_dst_cellId','current_dst_cellId','duration','size'}


def build_one(events_path, processed, splits, output):
    scenario = events_path.parent.relative_to(processed).as_posix()
    outdir = output / scenario
    outdir.mkdir(parents=True, exist_ok=True)
    nodes = {}
    counts = Counter()
    def node(kind, raw):
        raw = (raw or '').strip()
        if not raw or raw.lower() in {'none','nan','null','-','unknown'}:
            return None
        key = (kind, raw)
        if key not in nodes:
            nodes[key] = f'{kind}:{len(nodes)}'
            counts['node_'+kind] += 1
        return nodes[key]

    def edge(writer, src, dst, relation, event_id, timestamp):
        if src is not None and dst is not None:
            writer.writerow(dict(source=src,target=dst,relation=relation,
                                 event_id=event_id,timestamp=timestamp))
            counts['edge_'+relation] += 1

    with events_path.open(encoding='utf-8-sig',newline='') as f, \
         (outdir/'edges.csv').open('w',encoding='utf-8',newline='') as ef, \
         (outdir/'event_attributes.csv').open('w',encoding='utf-8',newline='') as af:
        reader = csv.DictReader(f)
        if not REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError(f'{events_path}: missing {sorted(REQUIRED-set(reader.fieldnames or []))}')
        ew = csv.DictWriter(ef,fieldnames=EDGE_FIELDS); ew.writeheader()
        aw = csv.DictWriter(af,fieldnames=EVENT_FIELDS); aw.writeheader()
        for idx, r in enumerate(reader):
            if r['scenario_id'] != scenario:
                raise ValueError(f'{events_path}: scenario_id mismatch at row {idx+2}')
            t = r['timestamp']
            ev = node('event', str(idx))
            # Use the actual node identifier for both event attributes and edge metadata.
            # A constructed f"event:{idx}" is NOT the same as the node ID, because
            # subscriber, device, and cell nodes also increment the node counter.
            event_id = ev
            src = node('subscriber',r['src_number'])
            dst = node('subscriber',r['dst_number'])
            aw.writerow(dict(event_id=event_id,timestamp=t,CDR_type=r['CDR_type'],
                             duration=r['duration'],size=r['size']))
            # A CDR is an event, not necessarily a person-to-person call.
            # Preserve its participants and attributes without inventing a call edge
            # for self-referential DATA traffic.
            edge(ew,src,ev,'originates_event',event_id,t)
            if r['CDR_type'] != 'DATA':
                edge(ew,ev,dst,'event_targets',event_id,t)
            for role, sub, prefix in [('src',src,'src'),('dst',dst,'dst')]:
                # Device and cell associations are observed within this event.
                if role == 'dst' and r['CDR_type'] == 'DATA':
                    continue
                imei = node('device',r[f'{prefix}_imei'])
                edge(ew,sub,imei,'uses_device',event_id,t)
                for field,relation in [('initial','initial_cell'),('current','current_cell')]:
                    cell = node('cell',r[f'{field}_{prefix}_cellId'])
                    edge(ew,sub,cell,relation,event_id,t)
            counts['events'] += 1

    with (outdir/'nodes.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=NODE_FIELDS); w.writeheader()
        for (kind,raw),identifier in nodes.items():
            w.writerow(dict(node_id=identifier,node_type=kind,raw_id=raw))

    target_count=0; fraud_count=0
    with (outdir/'targets.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=TARGET_FIELDS); w.writeheader()
        for subscriber_id,label,split in splits.get(scenario,[]):
            key=('subscriber',subscriber_id)
            if key not in nodes:
                raise ValueError(f'{scenario}: target subscriber absent from events: {subscriber_id}')
            w.writerow(dict(node_id=nodes[key],subscriber_id=subscriber_id,label=label,split=split))
            target_count+=1; fraud_count+=int(label)
    summary=dict(scenario_id=scenario,events=counts['events'],nodes=len(nodes),
                 edges=sum(v for k,v in counts.items() if k.startswith('edge_')),
                 node_types={k[5:]:v for k,v in counts.items() if k.startswith('node_')},
                 edge_types={k[5:]:v for k,v in counts.items() if k.startswith('edge_')},
                 targets=target_count,fraud_targets=fraud_count)
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(f"{scenario}: events={summary['events']:,}, nodes={summary['nodes']:,}, "
          f"edges={summary['edges']:,}, targets={target_count:,}, fraud={fraud_count:,}")
    return summary


def main():
    p=argparse.ArgumentParser(description='Build SIMGuard per-scenario graphs')
    p.add_argument('--processed',type=Path,default=Path('data/processed'))
    p.add_argument('--splits',type=Path,default=Path('data/splits/subscriber_splits.csv'))
    p.add_argument('--output',type=Path,default=Path('data/graphs'))
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.limit is not None and args.limit < 1: p.error('--limit must be >= 1')
    by_scenario={}
    with args.splits.open(encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f)
        if not {'scenario_id','subscriber_id','label','split'}.issubset(r.fieldnames or []):
            p.error('Unexpected split manifest columns')
        for row in r:
            by_scenario.setdefault(row['scenario_id'],[]).append(
                (row['subscriber_id'],row['label'],row['split']))
    paths=sorted(args.processed.rglob('events.csv'))
    if not paths: p.error(f'No events.csv files found in {args.processed}')
    if args.limit: paths=paths[:args.limit]
    summaries=[build_one(path,args.processed,by_scenario,args.output) for path in paths]
    print(f"Completed {len(summaries)} scenario graphs; "
          f"{sum(x['events'] for x in summaries):,} events total. Output: {args.output}")

if __name__=='__main__': main()
