"""Linux-only cloud batch: reverify, benchmark 2/4 workers, then four 100 EFC runs.

No AWS API, credentials, upload, GPU, or power-off logic is used here. The EC2
bootstrap supplies the independent shutdown deadline; outputs remain on its EBS.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CURRENTS = [3.0, 3.1, 3.2, 3.3]


def atomic_json(path, obj):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def meminfo_mb():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value, *_ = line.split()
        if key in ('MemTotal:', 'MemAvailable:'):
            values[key.rstrip(':')] = float(value) / 1024
    return values


def process_tree_mb(roots):
    """Count both run_100efc watchdog parents and their solver children."""
    procs = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            d = {}
            for line in (p / 'status').read_text().splitlines():
                if line.startswith(('PPid:', 'VmRSS:')):
                    key, value, *_ = line.split()
                    d[key] = int(value)
            procs[int(p.name)] = (d.get('PPid:', 0), d.get('VmRSS:', 0) / 1024)
        except (OSError, ValueError):
            pass
    included = set(roots)
    while True:
        children = {pid for pid, (parent, _) in procs.items() if parent in included}
        new = included | children
        if new == included:
            break
        included = new
    return sum(procs.get(pid, (0, 0))[1] for pid in included)


def stop_process(p):
    if p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
    except ProcessLookupError:
        pass


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def make_job(out, current, efc):
    name = f'current_{current:.1f}A'
    path = out / name
    cmd = [sys.executable, str(ROOT/'simulation/run_100efc.py'), '--cell', '4',
           '--current', str(current), '--mesh-scale', '4', '--efc', str(efc),
           '--output-dir', str(path), '--max-wall-s', '43200']
    return {'name': name, 'current_A': current, 'target_efc': efc,
            'output_dir': str(path), 'command': cmd}


def run_jobs(jobs, workers, stage, out, deadline):
    out.mkdir(parents=True, exist_ok=False)
    pending = deque(jobs)
    active = {}
    records = []
    started = time.perf_counter()
    last_progress = 0.0
    peak = 0.0
    total_mem = meminfo_mb()['MemTotal']
    try:
        while pending or active:
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Batch deadline reached during {stage}; last whole-cycle checkpoints retained')
            while pending and len(active) < workers:
                job = pending.popleft()
                log = (out / (job['name'] + '.log')).open('wb')
                env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                p = subprocess.Popen(job['command'], cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
                active[p.pid] = (p, log, job, time.perf_counter())
            rss = process_tree_mb(active)
            peak = max(peak, rss)
            available = meminfo_mb()['MemAvailable']
            if rss > .85 * total_mem or available < .08 * total_mem:
                raise MemoryError(f'Batch stopped before memory exhaustion: treeRSS={rss:.0f} MB, available={available:.0f} MB')
            for pid, (p, log, job, job_started) in list(active.items()):
                code = p.poll()
                if code is None:
                    continue
                log.close()
                summary = read_json(Path(job['output_dir'])/'summary.json')
                reached = bool(code == 0 and summary and
                    summary.get('terminal') and (Path(job['output_dir'])/'DONE').is_file() and
                    abs(summary['ledger']['efc']-job['target_efc']) <= 1e-8)
                record = {k: v for k, v in job.items() if k != 'command'}
                record.update({'exit_code': code, 'target_reached': reached,
                               'wall_s': time.perf_counter()-job_started,
                               'efc_reached': summary['ledger']['efc'] if summary else None})
                records.append(record)
                del active[pid]
                atomic_json(out/'progress.json', records)
                print(json.dumps({'stage':stage, 'finished':job['name'], 'target_reached':reached,
                                  'efc':record['efc_reached'], 'exit':code}), flush=True)
                if not reached:
                    raise RuntimeError(f"{stage}/{job['name']} failed; see its log and checkpoint")
            if time.monotonic()-last_progress >= 60:
                status = []
                for _, _, job, _ in active.values():
                    s = read_json(Path(job['output_dir'])/'summary.json')
                    status.append({'current_A':job['current_A'], 'efc':s['ledger']['efc'] if s else 0})
                print(json.dumps({'stage':stage, 'running':status, 'tree_rss_mb':round(rss,1),
                                  'finished':len(records)}), flush=True)
                last_progress = time.monotonic()
            if pending or active:
                time.sleep(1)
    finally:
        for p, log, _, _ in active.values():
            stop_process(p)
            log.close()
    elapsed = time.perf_counter()-started
    result = {'stage':stage, 'workers':workers, 'wall_s':elapsed, 'peak_process_tree_rss_mb':peak,
              'completed_efc':sum(r['efc_reached'] for r in records), 'jobs':records}
    result['completed_efc_per_hour'] = result['completed_efc'] / elapsed * 3600
    atomic_json(out/'stage_summary.json', result)
    return result


def run_verification(out, deadline):
    logpath = out/'linux_smoke.log'
    with logpath.open('wb') as log:
        cmd = [sys.executable, str(ROOT/'simulation/verify_100efc_runner.py'),
               '--output-dir', str(out/'linux_smoke'), '--mesh-scale', '1']
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        try:
            code = p.wait(timeout=min(900, max(1, deadline-time.monotonic())))
        except subprocess.TimeoutExpired:
            stop_process(p)
            raise RuntimeError('Linux integration smoke timed out')
    report = read_json(out/'linux_smoke/report.json')
    if code or not report or not report.get('all_pass'):
        raise RuntimeError('Linux integration smoke failed; Windows results do not certify Linux')
    print('LINUX_SMOKE_PASS', flush=True)
    return report


def compact_result(batch):
    results = []
    for job in batch.get('main', {}).get('jobs', []):
        s = read_json(Path(job['output_dir'])/'summary.json')
        if not s:
            continue
        times = [c['charge_time_s']/60 for c in s['cycles'] if 'charge_time_s' in c]
        results.append({'current_A':job['current_A'], 'efc':s['ledger']['efc'],
            'Q0_Ah':s['ledger']['q0_Ah'], 'diagnostics':[
                {'efc':d['efc'], 'SOH_percent':d['SOH_percent'], 'Qref_Ah':d['Qref_Ah']}
                for d in s['diagnostics']],
            'charge_min_minutes':min(times) if times else None,
            'charge_max_minutes':max(times) if times else None,
            'time_limit_violations':len(s['ledger']['time_violations']),
            'worker_peak_rss_mb':s['memory'].get('peak_rss_mb'),
            'elapsed_wall_s':s['elapsed_wall_s'], 'rows':s['row_count']})
    return {'status':batch['status'], 'selected_workers':batch.get('selected_workers'),
            'benchmark':[{k:v for k,v in x.items() if k != 'jobs'} for x in batch.get('benchmarks',[])],
            'results':results, 'error':batch.get('error'),
            'note':'First100 EFC only; not proof of EOL lifetime extension. Raw data remain on EBS.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--batch-max-wall-s', type=float, default=39600)
    p.add_argument('--skip-production', action='store_true', help='Only smoke and worker benchmarks')
    args = p.parse_args()
    if sys.platform != 'linux':
        p.error('This batch supervisor requires Linux /proc and process groups')
    if not 0 < args.batch_max_wall_s <= 39600:
        p.error('Batch wall limit must be <=11 hours; EC2 has a separate12hour shutdown')
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deadline = time.monotonic()+args.batch_max_wall_s
    batch = {'status':'running', 'started_utc':datetime.now(timezone.utc).isoformat(),
             'output_dir':str(out), 'platform':platform.platform(), 'python':platform.python_version(),
             'cpu_count':os.cpu_count(), 'memory_mb':meminfo_mb(), 'benchmarks':[],
             'versions':{n:importlib.metadata.version(n) for n in
                ['pybamm','pybammsolvers','casadi','numpy','scipy','pyarrow']}}
    exit_code = 0
    try:
        if shutil.disk_usage(out).free < 6*2**30:
            raise RuntimeError('At least6GiB free disk space required before batch')
        batch['linux_smoke'] = run_verification(out, deadline)
        for workers in [2,4]:
            if workers == 4 and batch['memory_mb']['MemTotal'] < 24*1024:
                break
            stage = out/f'benchmark_workers_{workers}'
            jobs = [make_job(stage,current,.05) for current in CURRENTS]
            bench = run_jobs(jobs,workers,f'benchmark_{workers}',stage,deadline)
            batch['benchmarks'].append(bench)
            atomic_json(out/'batch_summary.json',batch)
        first = batch['benchmarks'][0]
        chosen = 2
        if len(batch['benchmarks']) == 2:
            four = batch['benchmarks'][1]
            if four['wall_s'] < .95*first['wall_s'] and four['peak_process_tree_rss_mb'] < .75*batch['memory_mb']['MemTotal']:
                chosen = 4
        batch['selected_workers'] = chosen
        print('SELECTED_WORKERS='+str(chosen), flush=True)
        if not args.skip_production:
            stage = out/'main_100efc'
            batch['main'] = run_jobs([make_job(stage,i,100.) for i in CURRENTS],chosen,'main_100efc',stage,deadline)
        batch['status'] = 'completed_benchmark_only' if args.skip_production else 'completed'
    except Exception as e:
        exit_code = 1
        batch['status'] = 'failed_or_interrupted'
        batch['error'] = f'{type(e).__name__}: {e}'
        # Preserve partial per-trajectory summaries when a batch deadline stops workers.
        progress = read_json(out/'main_100efc/progress.json')
        jobs = progress or []
        known = {j['name'] for j in jobs}
        for current in CURRENTS:
            candidate = make_job(out/'main_100efc',current,100.)
            s = read_json(Path(candidate['output_dir'])/'summary.json')
            if s and candidate['name'] not in known:
                jobs.append({k:v for k,v in candidate.items() if k!='command'})
        if jobs:
            batch['main'] = {'jobs':jobs}
    finally:
        batch['wall_s'] = time.perf_counter()-started
        atomic_json(out/'batch_summary.json',batch)
        compact = compact_result(batch)
        atomic_json(out/'console_summary.json',compact)
        from write_cloud_report import write_report
        write_report(out,batch,compact)
        print('CAPSTONE_RESULT_BEGIN',flush=True)
        print(json.dumps(compact,ensure_ascii=False,separators=(',',':')),flush=True)
        print('CAPSTONE_RESULT_END',flush=True)
        print('REPORT_PATH='+str(out/'experiment_report.md'),flush=True)
    raise SystemExit(exit_code)


if __name__=='__main__':
    main()
