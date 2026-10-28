"""Pure lifetime policy and a frozen, reviewed legacy import profile."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
LEGACY_COMMIT='777153d2f8b695fa12510dcd50ac5d3ec16249c8'
LEGACY_SHA256={
 'configs/cells_pilot9.json':'f5e8866260ec5c857d8142559dd645d03916592c325eaa9dcb815c7c8e2aa4fc',
 'configs/diagnostics.json':'94cdd8bf5740faf03f587ff6e236b72df7131a067a4c90eed222f87e65aa8805',
 'configs/model.json':'acca23b6854b0942fbc2e6ca9a23045704bd469f5de22d1cad3ad9b4640b9090',
 'simulation/battery.py':'9de76603cef89fd247771d3983deb442901f9c17a677744ee522440813e96d3f',
 'simulation/run_100efc.py':'103e01a5c5cb1107aab04b7252e4578be16ed49c29f6274332fa6d25dbfbe759'}
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
def diagnostic_schedule(target, every_after_100=10.):
 if not (math.isfinite(target) and 0<target<=3000): raise ValueError('0 < target EFC <= 3000 required')
 if not (math.isfinite(every_after_100) and 0<every_after_100<=100): raise ValueError('0 < RPT interval <= 100 EFC required')
 points=[0.,5.,*map(float,range(10,101,10))]
 k=1
 while 100+k*every_after_100 < target-1e-9:
  points.append(100+k*every_after_100);k+=1
 points.append(float(target))
 return sorted({x for x in points if x<=target+1e-9})
def eol_bracket(diagnostics, threshold=80.):
 for i,d in enumerate(diagnostics):
  if d['SOH_percent']<=threshold:
   return {'threshold_percent':threshold,'observed_at_EFC':d['efc'],
    'interval_lower_EFC':diagnostics[i-1]['efc'] if i else None,
    'interval_upper_EFC':d['efc'],'exact_crossing_known':False,
    'rul_from_100_observation_EFC':d['efc']-100 if d['efc']>=100 else None}
 return None

def validate_legacy_identity(identity, root, *, cell,current,mesh,versions,python):
 expected={'cell':cell,'current_A':current,'mesh_scale':mesh,'output_period_s':1.,
           'versions':versions,'python':python,'tolerance':{'rtol':1e-4,'atol':1e-6}}
 for k,v in expected.items():
  if identity.get(k)!=v: raise ValueError('Legacy identity mismatch: '+k)
 if not (0<identity.get('efc_target',0)<=100): raise ValueError('Legacy target outside first-100 profile')
 hashes={k.replace('\\','/'):v for k,v in identity.get('source_sha256',{}).items()}
 if hashes!=LEGACY_SHA256: raise ValueError('Legacy checkpoint is not the reviewed source profile')
 for rel,expected_hash in LEGACY_SHA256.items():
  if digest(Path(root)/rel)!=expected_hash: raise ValueError('Current physical source differs from legacy: '+rel)
 return True
