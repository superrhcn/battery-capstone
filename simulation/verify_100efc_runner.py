"""Small integration verification for the 100 EFC runner, not a life experiment."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import battery as B
import run_100efc as R


def command(out, resume=False, mesh=1):
    cmd = [sys.executable, str(Path(R.__file__)), '--cell', '4', '--current', '3.3',
           '--mesh-scale', str(mesh), '--output-dir', str(out), '--smoke-cycle', '--max-wall-s', '600']
    if resume:
        cmd.append('--resume')
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=650)
    print(p.stdout, flush=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-5000:])


def compare(c, a, b, q0):
    sa, sb = B.last_step(a), B.last_step(b)
    da, db = c.degradation_Ah(sa), c.degradation_Ah(sb)
    return {'soc_diff': abs(c.soc(sa)-c.soc(sb)),
            'voltage_diff_V': abs(B.scalar(sa, B.VARS['voltage'])-B.scalar(sb, B.VARS['voltage'])),
            'degradation_max_diff_relative_Q0': max(abs(da[k]-db[k])/q0 for k in ['sei_Ah','plated_Ah','dead_Ah'])}


def passes(d):
    return d['soc_diff'] <= 1e-4 and d['voltage_diff_V'] <= 1e-3 and d['degradation_max_diff_relative_Q0'] <= 1e-4


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir', required=True)
    p.add_argument('--mesh-scale', type=int, choices=[1,4], default=1)
    args=p.parse_args()
    out=Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Verification output must be a new directory')
    out.mkdir(parents=True, exist_ok=True)
    start=time.perf_counter()
    mesh=args.mesh_scale
    report={'mesh_scale':mesh, 'all_pass':False}
    try:
        # Two distinct Python invocations save -> load -> continue from a whole-cycle checkpoint.
        command(out/'resume',mesh=mesh)
        first=R.load_checkpoint(out/'resume/checkpoint.pkl')
        command(out/'resume',resume=True,mesh=mesh)
        resumed=R.load_checkpoint(out/'resume/checkpoint.pkl')
        cell=B.Cell(mesh_scale=mesh)
        initial=cell.initial_state()
        q0=resumed['ledger']['q0_Ah']
        rs=B.RunState(4,'continuous_control',q0_Ah=q0,t_abs_s=float(initial.t[-1]))
        continuous,cycles,status,checks=B.run_operation(cell,initial,rs,3.3,5.,60.,max_cycles=2,
                                                       output_period_s=1.0,charge_limit_s=3600)
        comparison=compare(cell,continuous,resumed['solution'],q0)
        comparison['discharge_relative_Q0']=abs(rs.discharge_Ah-resumed['ledger']['discharge_Ah'])/q0
        comparison['time_diff_s']=abs(rs.t_abs_s-resumed['ledger']['t_abs_s'])
        report['fresh_process_resume']=comparison
        report['fresh_process_resume']['pass']=passes(comparison) and comparison['discharge_relative_Q0']<=1e-3
        # An internal diagnostic boundary must resume DIS, not restart CHARGE.
        rb=B.RunState(4,'uninterrupted',q0_Ah=q0,t_abs_s=float(initial.t[-1]))
        base,bc,bs,bcheck=B.run_operation(cell,initial,rb,3.3,5.,60.,efc_stop=.3,output_period_s=1.)
        cd=B.Cell(mesh_scale=mesh)
        sd=cd.initial_state()
        rd=B.RunState(4,'diagnostic_control',q0_Ah=q0,t_abs_s=float(sd.t[-1]))
        markers=[]
        def diag(sol,ledger,target):
            y=B.y_end(sol); old=copy.deepcopy(B.ledger(ledger))
            rpt=cd.reference_test(sol.last_state.copy())
            markers.append({'target':target,'actual':ledger.efc,'ok':rpt['ok'],
                            'state_unchanged':np.array_equal(y,B.y_end(sol)),
                            'ledger_unchanged':old==B.ledger(ledger), 'phase':ledger.stage})
        diagnostic,dc,ds,dcheck=B.run_operation(cd,sd,rd,3.3,5.,60.,efc_stop=.3,output_period_s=1.,
                                      diagnostic_efcs=[.1,.2,.3],on_diagnostic=diag)
        delta=compare(cd,base,diagnostic,q0)
        report['internal_diagnostic_resume']={'comparison':delta,'markers':markers,
            'pass':passes(delta) and ds=='efc_target' and len(markers)==3
                   and all(m['ok'] and m['state_unchanged'] and m['ledger_unchanged']
                           and m['phase']=='DISCHARGE' and abs(m['actual']-m['target'])<1e-9 for m in markers)
                   and len(dc)==len(bc)==1}
        # Re-read the saved parquet, integrate each nonzero-current piece, and compare ledger.
        total_dis=total_chg=0.; rows=0; largest_gap=0.; nonfinite=0
        for name in resumed['files']:
            data=pq.read_table(out/'resume'/name).to_pydict()
            t=np.asarray(data['time_s']); i=np.asarray(data['current_A']); pieces=np.asarray(data['piece'])
            rows+=len(t)
            nonfinite+=sum(not np.all(np.isfinite(np.asarray(data[k]))) for k in
                           ['time_s','current_A','voltage_V','soc_fraction','efc'])
            for piece in np.unique(pieces):
                select=pieces==piece; tt=t[select]; ii=i[select]
                largest_gap=max(largest_gap,float(np.diff(tt).max(initial=0)))
                total_dis+=float(np.trapezoid(np.maximum(ii,0),tt)/3600)
                total_chg+=float(np.trapezoid(np.maximum(-ii,0),tt)/3600)
        saved=resumed['ledger']
        dis_err=abs(total_dis-saved['discharge_Ah'])/q0
        chg_err=abs(total_chg-saved['charge_Ah'])/q0
        report['parquet_roundtrip']={'rows':rows,'row_count_matches':rows==resumed['row_count'],
            'maximum_within_piece_gap_s':largest_gap,'nonfinite_columns':nonfinite,
            'discharge_diff_relative_Q0':dis_err,'charge_diff_relative_Q0':chg_err,
            'pass':rows==resumed['row_count'] and nonfinite==0 and largest_gap<=1.+1e-8
                   and max(dis_err,chg_err)<=1e-3}
        report['all_pass']=all(report[k]['pass'] for k in
            ['fresh_process_resume','internal_diagnostic_resume','parquet_roundtrip'])
        report['wall_s']=time.perf_counter()-start
        R.save_json(out/'report.json',report)
        print(json.dumps(report,indent=2),flush=True)
        if not report['all_pass']:
            raise SystemExit(1)
    except Exception:
        R.save_json(out/'report.json',report)
        raise

if __name__=='__main__':
    main()
