"""Resumable phase controller. Uses the unchanged battery.py physical models."""
from __future__ import annotations
import copy
import numpy as np
import battery as B

def new_cursor(rs):
 return {'phase':'CHARGE_CC','cycle':None,'b':None,'prev_kind':None,
         'rest_end_t':None,'discharge_start_Ah':None}

def legacy_discharge_cursor(state):
 last=copy.deepcopy(state['cycles'][-1])
 if last.get('discharge_end')!='efc_target':
  raise ValueError('Only a terminal EFC stop inside DISCHARGE can be imported')
 for k in ('discharge_end','discharge_Ah','soc_discharge_end','soc_discharge_error'):
  last.pop(k,None)
 return {'phase':'DISCHARGE','cycle':last,'b':None,'prev_kind':'DIS',
         'rest_end_t':None,'discharge_start_Ah':state['ledger']['discharge_Ah']-state['cycles'][-1]['discharge_Ah']}

def run_segment(cell,sol,rs,cursor,current,target,diagnostics,on_diagnostic,on_piece,guard,
                *,charge_limit_s=3600.,output_period_s=1.):
 """Advance to a completed cycle or a controlled stop; keep partial phase cursor.

 guard -> None or stop reason; callback on_diagnostic -> None or stop reason.
 Never restarts charge when resuming a partial discharge.
 """
 ctx=cursor
 checks={'nonfinite_pieces':0,'time_join_max_error_s':0.,'transition_jump_max_rel':0.,'physical_join_max_rel':0.,'pieces_advanced':0,
         'min_values':{},'porosity_range':[1.,0.],'max_soc_error_fraction':0.}
 next_diags=[x for x in diagnostics if x>rs.efc+1e-9]
 next_diag=next_diags[0] if next_diags else None
 dt=float(B.MODEL_CFG['boundary_update_s'])
 rest=float(B.MODEL_CFG['rest_s'])
 dis=float(B.MODEL_CFG['currents_A']['discharge'])
 if ctx['b'] is None:
  ctx['b']=cell.boundaries(B.last_step(sol));rs.next_boundary_t_s=rs.t_abs_s+dt

 def boundary():
  ctx['b']=cell.boundaries(B.last_step(sol));rs.next_boundary_t_s=rs.t_abs_s+dt
 def enter(phase):
  ctx['phase']=phase;rs.stage=phase;boundary()
  if phase.startswith('REST'):
   ctx['rest_end_t']=rs.t_abs_s+rest;rs.rest_remaining_s=rest
  if phase=='DISCHARGE': ctx['discharge_start_Ah']=rs.discharge_Ah
 def soc(): return B.soc_from_x(B.scalar(B.last_step(sol),B.X),ctx['b'])
 def finish_charge(end):
  boundary();s=soc()
  ctx['cycle'].update(charge_end=end,cv_entered=rs.cv_start_s is not None,
   charge_time_s=rs.t_abs_s-rs.charge_start_s,
   cv_time_s=rs.t_abs_s-rs.cv_start_s if rs.cv_start_s is not None else 0.,
   soc_charge_end=s,soc_charge_error=s-B.SOC_HI)
  if abs(s-B.SOC_HI)>B.SOC_TOL: raise RuntimeError('Charge SOC error exceeds operating tolerance')
  checks['max_soc_error_fraction']=max(checks['max_soc_error_fraction'],abs(s-B.SOC_HI))
  if ctx['cycle']['charge_time_s']>charge_limit_s:
   rs.time_violations.append({'cycle':ctx['cycle']['cycle'],'charge_time_s':ctx['cycle']['charge_time_s']})
  enter('REST_AFTER_CHARGE')
 def finish_discharge(end):
  boundary();s=soc()
  ctx['cycle'].update(discharge_end=end,discharge_Ah=rs.discharge_Ah-ctx['discharge_start_Ah'],
                      soc_discharge_end=s,soc_discharge_error=s-B.SOC_LO)
  if abs(s-B.SOC_LO)>B.SOC_TOL: raise RuntimeError('Discharge SOC error exceeds operating tolerance')
  checks['max_soc_error_fraction']=max(checks['max_soc_error_fraction'],abs(s-B.SOC_LO))
  enter('REST_AFTER_DISCHARGE')

 while True:
  reason=guard()
  if reason: return sol,ctx,reason,checks,None
  if rs.efc>=target-1e-9: return sol,ctx,'max_efc_censored',checks,None
  phase=ctx['phase'];rs.stage=phase
  if ctx['cycle'] is None:
   ctx['cycle']={'cycle':rs.cycle+1,'efc_start':rs.efc}
   rs.charge_start_s=rs.t_abs_s;rs.cv_start_s=None
   boundary()
  if phase.startswith('REST') and rs.t_abs_s>=ctx['rest_end_t']-1e-9:
   rs.rest_remaining_s=0.
   if phase=='REST_AFTER_CHARGE': enter('DISCHARGE');continue
   completed=copy.deepcopy(ctx['cycle']);rs.cycle+=1
   ctx=new_cursor(rs);rs.stage=ctx['phase'];rs.next_boundary_t_s=rs.t_abs_s
   return sol,ctx,'cycle_complete',checks,completed
  if rs.t_abs_s>=rs.next_boundary_t_s-1e-9: boundary()
  b=ctx['b'];s=soc()
  if phase in ('CHARGE_CC','CHARGE_CV') and s>=B.SOC_HI:
   finish_charge('soc_already');continue
  if phase=='DISCHARGE' and s<=B.SOC_LO:
   finish_discharge('soc_already');continue
  limit=rs.next_boundary_t_s
  if phase.startswith('REST'): limit=min(limit,ctx['rest_end_t'])
  if phase=='DISCHARGE':
   limit=min(limit,rs.t_abs_s+(target*rs.q0_Ah-rs.discharge_Ah)/dis*3600.)
   if next_diag is not None:
    limit=min(limit,rs.t_abs_s+(next_diag*rs.q0_Ah-rs.discharge_Ah)/dis*3600.)
  if limit-rs.t_abs_s<=1e-10: raise RuntimeError('Controller made no progress')
  inputs={B.T_STOP:limit}
  if phase in ('CHARGE_CC','CHARGE_CV'):
   kind='CC' if phase=='CHARGE_CC' else 'CV'
   step=B.step_cc(current,dt,output_period_s) if kind=='CC' else B.step_cv(dt,output_period_s)
   inputs[B.X_HI]=B.x_from_soc(B.SOC_HI,b)
  elif phase=='DISCHARGE':
   kind='DIS';step=B.step_dis(dis,dt,output_period_s);inputs[B.X_LO]=B.x_from_soc(B.SOC_LO,b)
  else: kind='REST';step=B.step_rest(dt,output_period_s)
  prev=B.last_step(sol)
  key=(kind,current if kind in ('CC','CV') else dis if kind=='DIS' else 0.,dt,output_period_s)
  new,st,wall=cell.solve(key,[step],sol.last_state,inputs)
  checks['time_join_max_error_s']=max(checks['time_join_max_error_s'],abs(float(st.t[0])-rs.t_abs_s))
  if checks['time_join_max_error_s']>1e-6: raise RuntimeError('Time connection mismatch')
  y=np.asarray(st.all_ys[-1],dtype=float)
  if not np.all(np.isfinite(y)): raise RuntimeError('Nonfinite solver state')
  dif=B.snap_diff(B.snapshot(prev,-1),B.snapshot(st,0))
  if any('status' in v for v in dif.values()): raise RuntimeError('Physical state shape mismatch')
  jump=max((v.get('max_rel_diff',0.) for k,v in dif.items() if k!='discharge_capacity'),default=0.)
  checks['physical_join_max_rel']=max(checks['physical_join_max_rel'],jump)
  if ctx['prev_kind'] and ctx['prev_kind']!=kind:
   checks['transition_jump_max_rel']=max(checks['transition_jump_max_rel'],jump)
  if jump>1e-10: raise RuntimeError('Physical state connection mismatch')
  checks['pieces_advanced']+=1
  # Derived physical constraints at each piece, including the last partial segment.
  for k in ('plated_li','dead_li','sei_thickness','electrolyte_conc','neg_particle_conc','pos_particle_conc'):
   minimum=float(np.min(np.asarray(st[B.STATE_VARS[k]].entries,dtype=float)))
   checks['min_values'][k]=min(checks['min_values'].get(k,float('inf')),minimum)
   if minimum<0: raise RuntimeError('Negative physical variable: '+k)
  por=np.asarray(st[B.STATE_VARS['neg_porosity']].entries,dtype=float)
  checks['porosity_range']=[min(checks['porosity_range'][0],float(por.min())),max(checks['porosity_range'][1],float(por.max()))]
  if not (0<checks['porosity_range'][0]<=checks['porosity_range'][1]<1): raise RuntimeError('Porosity outside (0,1)')
  q=np.asarray(st[B.VARS['discharge_capacity']].entries,dtype=float);dq=float(q[-1]-q[0])
  if dq>=0: rs.discharge_Ah+=dq
  else: rs.charge_Ah-=dq
  rs.efc=rs.discharge_Ah/rs.q0_Ah;rs.t_abs_s=float(st.t[-1]);rs.wall_time_s+=wall;rs.pieces+=1
  sol=new;ctx['prev_kind']=kind
  if phase.startswith('REST'): rs.rest_remaining_s=max(0.,ctx['rest_end_t']-rs.t_abs_s)
  if on_piece: on_piece(sol,st,kind,b,rs)
  cls=B.classify(st)
  if float(st.t[-1]-st.t[0])<=1e-9: raise RuntimeError('Zero-duration solver piece')
  if kind=='DIS' and next_diag is not None and rs.efc>=next_diag-1e-9:
   reason=on_diagnostic(sol,rs,next_diag)
   next_diags.pop(0);next_diag=next_diags[0] if next_diags else None
   if reason: return sol,ctx,reason,checks,None
  if kind in ('CC','CV'):
   if cls=='soc': finish_charge(cls)
   elif cls=='voltage' and kind=='CC':
    rs.cv_start_s=rs.t_abs_s;enter('CHARGE_CV')
   elif cls=='time':
    if kind=='CV' and abs(B.scalar(st,B.VARS['current']))<1e-3: return sol,ctx,'operating_condition_not_met:cv_current_stall',checks,None
   else: return sol,ctx,'operating_condition_not_met:charge:'+cls,checks,None
  elif kind=='DIS':
   if cls=='soc': finish_discharge(cls)
   elif cls=='voltage': return sol,ctx,'operating_condition_not_met:early_voltage_cutoff',checks,None
   elif cls!='time': return sol,ctx,'operating_condition_not_met:discharge:'+cls,checks,None
  elif cls!='time': return sol,ctx,'operating_condition_not_met:rest:'+cls,checks,None
