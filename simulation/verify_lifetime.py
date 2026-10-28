"""Short policy and physical checks for the lifetime runner; never a life test."""
from __future__ import annotations
import argparse,copy,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import battery as B
import run_100efc as R
import run_lifetime as L
import lifetime_operation as O
import lifetime_policy as P

def run(cmd):
 p=subprocess.run([sys.executable,'-X','utf8',*map(str,cmd)],capture_output=True,text=True,encoding='utf-8',timeout=600)
 print(p.stdout,flush=True)
 if p.returncode:raise RuntimeError(p.stderr[-6000:])
def compare(cell,a,b,q0):
 a,b=B.last_step(a),B.last_step(b)
 da,db=cell.degradation_Ah(a),cell.degradation_Ah(b)
 return {'soc_fraction':abs(cell.soc(a)-cell.soc(b)),
         'voltage_V':abs(B.scalar(a,B.VARS['voltage'])-B.scalar(b,B.VARS['voltage'])),
         'degradation_relative_Q0':max(abs(da[k]-db[k])/q0 for k in ('sei_Ah','plated_Ah','dead_Ah'))}
def passing(c):return c['soc_fraction']<=1e-4 and c['voltage_V']<=1e-3 and c['degradation_relative_Q0']<=1e-4

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);a=p.parse_args()
 out=Path(a.output_dir).resolve()
 if out.exists() and any(out.iterdir()):raise ValueError('Use a new verification output directory')
 out.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
 report={'all_pass':False,'scope':'mesh1 short tests, not a life or mesh accuracy test'}
 try:
  schedule=P.diagnostic_schedule(3000)
  assert schedule[:12]==[0.,5.,10.,20.,30.,40.,50.,60.,70.,80.,90.,100.]
  assert schedule[-1]==3000 and len(schedule)==302
  for invalid in (0,-1,3001,float('nan'),float('inf')):
   try:P.diagnostic_schedule(invalid)
   except ValueError:pass
   else:raise AssertionError('Invalid target accepted')
  bracket=P.eol_bracket([{'efc':100,'SOH_percent':99},{'efc':200,'SOH_percent':80.1},{'efc':300,'SOH_percent':79.9}])
  assert bracket['interval_lower_EFC']==200 and bracket['interval_upper_EFC']==300 and not bracket['exact_crossing_known']
  assert P.eol_bracket([{'efc':100,'SOH_percent':99}]) is None
  report['policy']={'pass':True,'EOL_interval_example':bracket,'diagnostic_count_to_3000':len(schedule)}
  legacy=out/'legacy'
  run([Path(R.__file__),'--cell',4,'--current',3.2,'--mesh-scale',1,'--efc',.05,'--max-wall-s',600,'--output-dir',legacy])
  old=R.load_checkpoint(legacy/'checkpoint.pkl')
  assert old['terminal'] and old['cycles'][-1]['discharge_end']=='efc_target'
  original={f.name:P.digest(f) for f in legacy.iterdir() if f.is_file()}
  common=[Path(L.__file__),'--cell',4,'--current',3.2,'--mesh-scale',1,'--efc',1.1,'--max-wall-s',600]
  resumed=out/'resumed'
  run([*common,'--output-dir',resumed,'--import-100efc-dir',legacy,'--smoke-segments',1])
  paused=R.load_checkpoint(resumed/'checkpoint.pkl')
  assert paused['status']=='paused_verified' and paused['ledger']['cycle']==old['ledger']['cycle']+1
  first=pq.read_table(resumed/paused['files'][0]).to_pydict()
  assert first['phase'][0]=='DIS' and abs(first['time_s'][0]-old['ledger']['t_abs_s'])<1e-6
  run([*common,'--output-dir',resumed,'--resume'])
  rs=R.load_checkpoint(resumed/'checkpoint.pkl')
  assert rs['status']=='max_efc_censored' and abs(rs['ledger']['efc']-1.1)<1e-8
  control=out/'control'
  run([*common,'--output-dir',control,'--import-100efc-dir',legacy])
  base=R.load_checkpoint(control/'checkpoint.pkl');cell=B.Cell(mesh_scale=1)
  delta=compare(cell,rs['solution'],base['solution'],rs['ledger']['q0_Ah'])
  delta['time_s']=abs(rs['ledger']['t_abs_s']-base['ledger']['t_abs_s'])
  delta['EFC']=abs(rs['ledger']['efc']-base['ledger']['efc'])
  assert passing(delta) and delta['time_s']<1e-3 and delta['EFC']<1e-8
  assert original=={f.name:P.digest(f) for f in legacy.iterdir() if f.is_file()}
  rows=0;dis=0.;chg=0.;maxgap=0.
  for name in rs['files']:
   data=pq.read_table(resumed/name).to_pydict();rows+=len(data['time_s'])
   tt=np.asarray(data['time_s']);ii=np.asarray(data['current_A']);piece=np.asarray(data['piece'])
   for k in np.unique(piece):
    sel=piece==k;t=tt[sel];i=ii[sel];maxgap=max(maxgap,float(np.diff(t).max(initial=0)))
    dis+=float(np.trapezoid(np.maximum(i,0),t)/3600.);chg+=float(np.trapezoid(np.maximum(-i,0),t)/3600.)
  q0=rs['ledger']['q0_Ah'];diserr=abs(dis-(rs['ledger']['discharge_Ah']-old['ledger']['discharge_Ah']))/q0
  chgerr=abs(chg-(rs['ledger']['charge_Ah']-old['ledger']['charge_Ah']))/q0
  assert rows==rs['row_count'] and maxgap<=1.+1e-8 and max(diserr,chgerr)<=1e-3
  # Compare with the unchanged, previously verified physical controller as well.
  independent=B.Cell(mesh_scale=1);initial=independent.initial_state()
  old_rs=B.RunState(4,'legacy_control',q0_Ah=q0,t_abs_s=float(initial.t[-1]))
  old_sol,_,_,_=B.run_operation(independent,initial,old_rs,3.2,5.,60.,efc_stop=1.1,
       output_period_s=1.,diagnostic_efcs=[.05,1.1],on_diagnostic=lambda *_:None)
  old_delta=compare(independent,base['solution'],old_sol,q0)
  assert passing(old_delta)
  # Controlled stop paths must return a resumable DIS cursor, not start CHARGE.
  guard_rs=B.RunState(**copy.deepcopy(old['ledger']));guard_cursor=O.legacy_discharge_cursor(old)
  guard_start=old['solution'].last_state.copy();guard_y=B.y_end(guard_start)
  guarded,cursor,status,_,_=O.run_segment(cell,guard_start,guard_rs,guard_cursor,3.2,1.1,[],
       lambda *_:None,lambda *_:(_ for _ in ()).throw(AssertionError('Guard must stop before a piece')),lambda:'wall_time_censored')
  assert status=='wall_time_censored' and cursor['phase']=='DISCHARGE' and np.array_equal(guard_y,B.y_end(guarded))
  eol_rs=B.RunState(**copy.deepcopy(old['ledger']));eol_cursor=O.legacy_discharge_cursor(old)
  _,eol_cursor,eol_status,_,_=O.run_segment(cell,old['solution'].last_state.copy(),eol_rs,eol_cursor,3.2,1.1,[.1],
       lambda *_:'eol_observed',None,lambda:None)
  assert eol_status=='eol_observed' and abs(eol_rs.efc-.1)<1e-9 and eol_cursor['phase']=='DISCHARGE'
  report['unchanged_physics_controller_comparison']=old_delta
  report['controlled_stop_tests']={'timeout_before_piece':'pass','EOL_callback_stop_mid_DIS':'pass',
        'note':'EOL callback is a test sentinel, not a measured SOH80 crossing'}
  # A different old current/version/source must be rejected, rather than disabling identity checks.
  invalid=copy.deepcopy(old['identity']);invalid['current_A']=3.3
  try:P.validate_legacy_identity(invalid,B.ROOT,cell=4,current=3.2,mesh=1,versions=L.versions(),python=L.platform.python_version())
  except ValueError:pass
  else:raise AssertionError('Mismatched import accepted')
  report.update(all_pass=True,legacy_input_unchanged=True,first_extension_phase=first['phase'][0],
    imported_Q0_Ah=q0,fresh_process_resume=delta,extension_rows=rows,
    parquet={'max_gap_s':maxgap,'discharge_error_relative_Q0':diserr,'charge_error_relative_Q0':chgerr},
    terminal_status=rs['status'],target_EFC=rs['ledger']['efc'],wall_s=time.perf_counter()-started)
 finally:R.save_json(out/'report.json',report)
 print(json.dumps(report,indent=2),flush=True)
 if not report['all_pass']:raise SystemExit(1)
if __name__=='__main__':main()
