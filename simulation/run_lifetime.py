"""Single-cell lifetime experiment, independent of chat and ML training.

New output directory required for --import-100efc-dir. Legacy files are read-only.
Diagnostic EOL is an interval-censored crossing, not an exact crossing time.
"""
from __future__ import annotations
import argparse,copy,hashlib,importlib.metadata,json,math,os,platform,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ.setdefault(name,'1')
import numpy as np
import battery as B
import run_100efc as R
import lifetime_operation as O
import lifetime_policy as P
import lifetime_supervision as S

PACKAGES=('pybamm','pybammsolvers','casadi','numpy','scipy','pyarrow')
def versions():return {n:importlib.metadata.version(n) for n in PACKAGES}
def identity(args):
 files=[B.ROOT/rel for rel in P.LEGACY_SHA256]
 files += [Path(__file__),Path(O.__file__),Path(P.__file__),Path(S.__file__)]
 return {'format':'lifetime-v1','cell':args.cell,'current_A':args.current,'mesh_scale':args.mesh_scale,
  'target_EFC':args.efc,'eol_SOH_percent':args.eol_soh,'rpt_every_after_100_EFC':args.rpt_every_efc,
  'output_period_s':1.,'versions':versions(),'python':platform.python_version(),
  'source_sha256':{p.relative_to(B.ROOT).as_posix():P.digest(p) for p in files},
  'tolerance':{'rtol':B.DEFAULT_RTOL,'atol':B.DEFAULT_ATOL}}
def make_cell(args):
 specs=json.loads((B.ROOT/'configs/cells_pilot9.json').read_text(encoding='utf-8'))['cells']
 c=next(c for c in specs if c['id']==args.cell)
 return B.Cell({'Negative particle radius [m]':c['negative_particle_radius_m'],
                'SEI solvent diffusivity [m2.s-1]':c['sei_solvent_diffusivity_m2s']},mesh_scale=args.mesh_scale)
def import_legacy(args,out):
 source=Path(args.import_100efc_dir).resolve()
 if source==out or source in out.parents or out in source.parents:
  raise ValueError('Legacy input and new output must be disjoint directories')
 summary=json.loads((source/'summary.json').read_text(encoding='utf-8'))
 P.validate_legacy_identity(summary['identity'],B.ROOT,cell=args.cell,current=args.current,
  mesh=args.mesh_scale,versions=versions(),python=platform.python_version())
 names=['checkpoint.pkl','summary.json','diagnostics.json',*summary['files']]
 if any(Path(n).name!=n for n in names):raise ValueError('Unsafe legacy filename')
 before={n:P.digest(source/n) for n in names}
 old=R.load_checkpoint(source/'checkpoint.pkl')
 if old['identity']!=summary['identity'] or old['ledger']!=summary['ledger']:
  raise ValueError('Legacy checkpoint and JSON disagree')
 if not old.get('terminal') or old.get('status')!='efc_target':
  raise ValueError('Legacy import requires a completed EFC-target checkpoint')
 if abs(old['ledger']['efc']-old['identity']['efc_target'])>1e-8:
  raise ValueError('Legacy target and ledger disagree')
 if args.efc<=old['ledger']['efc']+1e-9:raise ValueError('New target must extend the old target')
 if P.eol_bracket(old['diagnostics'],args.eol_soh):raise ValueError('Legacy data already crossed requested EOL')
 cursor=O.legacy_discharge_cursor(old)
 if before!={n:P.digest(source/n) for n in names}:raise ValueError('Legacy input changed while reading')
 lineage={'source_commit':P.LEGACY_COMMIT,'source_dir':str(source),'sha256':before,
          'source_EFC':old['ledger']['efc'],'source_elapsed_wall_s':old['elapsed_wall_s'],
          'source_phase_inferred':'DISCHARGE','new_files_contain':'extension observations only',
          'source_files_unchanged':True}
 R.save_json(out/'lineage.json',lineage)
 return {'identity':identity(args),'ledger':old['ledger'],'solution':old['solution'].last_state,
  'cursor':cursor,'cycles':[],'diagnostics':copy.deepcopy(old['diagnostics']),
  'files':[],'row_count':0,'elapsed_wall_s':old['elapsed_wall_s'],
  'extension_elapsed_wall_s':0.,'source_elapsed_wall_s':old['elapsed_wall_s'],
  'lineage':lineage,'terminal':False,'status':'ready','checks':[]}
def summary_of(state,args,status=None):
 data={k:v for k,v in state.items() if k!='solution'}
 data.update(status=status or state['status'],memory=B.memory_mb(),max_wall_s=args.max_wall_s,
             max_EFC=args.efc,eol=P.eol_bracket(state['diagnostics'],args.eol_soh),
             resource_limits={'max_wall_s':args.max_wall_s,'max_memory_GiB':args.max_memory_gib,
                              'min_free_memory_GiB':args.min_free_memory_gib,'min_free_disk_GiB':args.min_free_disk_gib},
             diagnostic_schedule_EFC=P.diagnostic_schedule(args.efc,args.rpt_every_efc),
             completed_cycles_total=state['ledger']['cycle'],
             limitations=['EOL crossing is bracketed by RPT observations, not exact.',
                          'Partial extension data require legacy source data from lineage.',
                          'No ML training or current optimization in this runner.'])
 return data

def worker(args):
 out=Path(args.output_dir).resolve();cp=out/'checkpoint.pkl';sig=identity(args)
 if args.resume:
  state=R.load_checkpoint(cp)
  if state['identity']!=sig:raise RuntimeError('Lifetime checkpoint source/config/version identity mismatch')
  if state.get('terminal'):raise RuntimeError('Terminal run cannot be resumed')
  committed=set(state['files'])
  # Only uncommitted extension files in the dedicated output directory can be removed.
  for p in out.glob('segment_*.parquet*'):
   if p.name not in committed:p.unlink()
 elif args.import_100efc_dir:
  if out.exists() and any(out.iterdir()):raise RuntimeError('New lifetime output must be empty')
  out.mkdir(parents=True,exist_ok=True);state=import_legacy(args,out)
 else:
  if out.exists() and any(out.iterdir()):raise RuntimeError('New lifetime output must be empty')
  out.mkdir(parents=True,exist_ok=True)
  state={'identity':sig,'cycles':[],'diagnostics':[],'files':[],'row_count':0,'checks':[],
         'elapsed_wall_s':0.,'extension_elapsed_wall_s':0.,'source_elapsed_wall_s':0.,
         'terminal':False,'status':'ready','lineage':None}
 prior=float(state['elapsed_wall_s'])
 # A previous kill may have consumed work beyond the last safe state.
 for name in ('outcome.json','FAILED.json'):
  if args.resume and (out/name).is_file():
   previous=json.loads((out/name).read_text(encoding='utf-8'))
   prior=max(prior,float(previous.get('elapsed_wall_s',0)))
 start=time.perf_counter();cell=make_cell(args);obs=None
 if 'solution' in state:
  sol=state['solution'];rs=B.RunState(**state['ledger']);cursor=state['cursor']
 else:
  sol=cell.initial_state();rs=B.RunState(args.cell,f'cell{args.cell}_{args.current:g}A_mesh{args.mesh_scale:g}_lifetime',t_abs_s=float(sol.t[-1]))
  cursor=O.new_cursor(rs)
 stop_signal={'requested':False}
 def on_signal(*_):stop_signal['requested']=True
 if hasattr(signal,'SIGTERM'):signal.signal(signal.SIGTERM,on_signal)
 def elapsed():return prior+time.perf_counter()-start
 def guard():
  if stop_signal['requested']:return 'interrupted_censored'
  if elapsed()>=args.max_wall_s:return 'wall_time_censored'
  mem=B.memory_mb()
  if mem.get('rss_mb',0)>args.max_memory_gib*1024:return 'resource_limit_censored:process_memory'
  if shutil.disk_usage(out).free<args.min_free_disk_gib*2**30:return 'resource_limit_censored:disk'
  if sys.platform.startswith('linux'):
   available=next((float(x.split()[1])/1024**2 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')),None)
   if available is not None and available<args.min_free_memory_gib:return 'resource_limit_censored:system_memory'
  return None
 def diagnose(main,ledger,target):
  before=B.y_end(main);book=copy.deepcopy(B.ledger(ledger))
  result=cell.reference_test(main.last_state.copy());q=result['Q_cum_Ah']
  if not result['ok'] or not np.isfinite(q) or q<=0 or abs(result['Q_trapz_Ah']-q)/q>1e-4:
   raise RuntimeError('Invalid reference capacity')
  if not np.array_equal(before,B.y_end(main)) or book!=B.ledger(ledger):raise RuntimeError('Diagnostic modified main state')
  if ledger.q0_Ah is None:ledger.q0_Ah=q
  point={'efc':ledger.efc,'target_efc':target,'Qref_Ah':q,'SOH_percent':100*q/ledger.q0_Ah,
    'state_max_abs_change':0.,'ledger_unchanged':True,'reference':result,
    'output_period':'reference_test Experiment default'}
  state['diagnostics'].append(point)
  R.save_json(out/'diagnostics.json',state['diagnostics'])
  print(json.dumps({'event':'RPT','EFC':ledger.efc,'SOH_percent':point['SOH_percent']}),flush=True)
  return 'eol_observed' if point['SOH_percent']<=args.eol_soh else guard()
 def checkpoint(status,terminal=False):
  state.update(ledger=B.ledger(rs),solution=sol.last_state,cursor=cursor,
   elapsed_wall_s=elapsed(),extension_elapsed_wall_s=elapsed()-state['source_elapsed_wall_s'],
   terminal=terminal,status=status)
  R.save_checkpoint(cp,state);R.save_json(out/'summary.json',summary_of(state,args))
  R.save_json(out/'diagnostics.json',state['diagnostics'])
 def finish(status):
  terminal=status in ('eol_observed','max_efc_censored') or status.startswith('operating_condition_not_met')
  checkpoint(status,terminal)
  result=summary_of(state,args)
  result['right_censored']=status!='eol_observed'
  result['last_valid_EFC']=rs.efc
  R.save_json(out/'outcome.json',result)
  if terminal:(out/'DONE').write_text(status+'\n',encoding='utf-8')
  print(json.dumps({'status':status,'efc':rs.efc,'elapsed_wall_s':state['elapsed_wall_s'],'eol':result['eol']}),flush=True)
  return 0 if terminal or status=='paused_verified' else 124
 try:
  if not state['diagnostics']:
   reason=diagnose(sol,rs,0.)
   if reason:return finish(reason)
  checkpoint('running')
  targets=P.diagnostic_schedule(args.efc,args.rpt_every_efc)
  segments_this_invocation=0
  while True:
   reason=guard()
   if reason:return finish(reason)
   if rs.efc>=args.efc-1e-9:return finish('max_efc_censored')
   segment_id=len(state['files'])+1
   obs=R.Observations(out/f'segment_{segment_id:06d}.parquet')
   def on_piece(main,st,kind,b,ledger):obs.add(main,st,kind,b,ledger)
   sol,cursor,status,checks,completed=O.run_segment(cell,sol,rs,cursor,args.current,args.efc,
       targets,diagnose,on_piece,guard)
   obs.close()
   if obs.rows:
    state['files'].append(obs.path.name);state['row_count']+=obs.rows
   obs=None
   if completed:state['cycles'].append(completed)
   state['checks'].append(checks);segments_this_invocation+=1
   if status!='cycle_complete':return finish(status)
   checkpoint('running')
   print(json.dumps({'event':'cycle','cycle':rs.cycle,'EFC':rs.efc,'elapsed_wall_s':state['elapsed_wall_s'],'memory':B.memory_mb()}),flush=True)
   if args.smoke_segments and segments_this_invocation>=args.smoke_segments:return finish('paused_verified')
 except BaseException:
  if obs:obs.close(commit=False)
  # Preserve last validated checkpoint. Failure is not an observed battery EOL.
  R.save_json(out/'FAILED.json',{'status':'failed','error':traceback.format_exc(),
      'elapsed_wall_s':elapsed(),'last_safe_checkpoint':str(cp),'memory':B.memory_mb()})
  raise

def parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--cell',type=int,default=4,choices=range(1,10))
 p.add_argument('--current',type=float,default=3.2,choices=[3.,3.1,3.2,3.3])
 p.add_argument('--mesh-scale',type=int,default=4,choices=[1,2,4,8])
 p.add_argument('--efc',type=float,default=3000.)
 p.add_argument('--eol-soh',type=float,default=80.)
 p.add_argument('--rpt-every-efc',type=float,default=10.)
 p.add_argument('--max-wall-s',type=float,default=72000.)
 p.add_argument('--max-memory-gib',type=float,default=24.)
 p.add_argument('--min-free-memory-gib',type=float,default=2.)
 p.add_argument('--min-free-disk-gib',type=float,default=2.)
 p.add_argument('--output-dir',required=True)
 g=p.add_mutually_exclusive_group();g.add_argument('--resume',action='store_true');g.add_argument('--import-100efc-dir')
 p.add_argument('--smoke-segments',type=int,default=0,help='Verification only: pause after N saved segments')
 p.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
 return p

def main():
 p=parser();args=p.parse_args()
 try:P.diagnostic_schedule(args.efc,args.rpt_every_efc)
 except ValueError as e:p.error(str(e))
 if not all(math.isfinite(x) and x>0 for x in (args.max_wall_s,args.max_memory_gib,args.min_free_memory_gib,args.min_free_disk_gib)):p.error('Positive finite resource limits required')
 if args.max_wall_s>72000 or not (0<args.eol_soh<=100) or args.smoke_segments<0:p.error('20-hour maximum / valid EOL / nonnegative smoke count required')
 if args._worker:raise SystemExit(worker(args))
 out=Path(args.output_dir).resolve();out.parent.mkdir(parents=True,exist_ok=True)
 prior=0.
 source=out if args.resume else Path(args.import_100efc_dir) if args.import_100efc_dir else None
 if source:
  for name in ('summary.json','outcome.json','FAILED.json'):
   if (source/name).is_file():prior=max(prior,float(json.loads((source/name).read_text(encoding='utf-8')).get('elapsed_wall_s',0)))
 if prior>=args.max_wall_s:raise SystemExit('Cumulative 20-hour trajectory budget already exhausted')
 lock=out.parent/(out.name+'.runner.lock')
 parent_started=time.perf_counter()
 def forced_stop(reason):
  out.mkdir(parents=True,exist_ok=True)
  safe=json.loads((out/'summary.json').read_text(encoding='utf-8')) if (out/'summary.json').is_file() else {}
  R.save_json(out/'outcome.json',{'status':reason,'right_censored':True,
    'elapsed_wall_s':min(args.max_wall_s,prior+time.perf_counter()-parent_started),
    'last_valid_EFC':safe.get('ledger',{}).get('efc'),
    'message':'Worker killed after bounded shutdown wait; last checkpoint remains authoritative; partial file uncommitted'})
 code=S.supervise([sys.executable,__file__,*sys.argv[1:],'--_worker'],lock,
                  args.max_wall_s-prior,grace_s=60.,on_forced_stop=forced_stop)
 raise SystemExit(code)
if __name__=='__main__':main()
