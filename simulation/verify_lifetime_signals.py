"""Verify bounded parent/worker signal handling, without battery calculations.

On POSIX also sends real parent-only and process-group SIGTERM to fixtures.
Windows runs deterministic handler/policy tests and labels POSIX tests skipped.
"""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,tempfile,time
from pathlib import Path
from unittest.mock import patch
import lifetime_supervision as S

def policy_checks(root):
 events=[]
 class Fake:
  def __init__(self,*_,**__):self.code=None;self.forwarded=False;self.calls=0
  def poll(self):return self.code
  def send_signal(self,sig):self.forwarded=True;events.append('forwarded')
  def terminate(self):self.send_signal(signal.SIGTERM)
  def kill(self):self.code=-9;events.append('killed')
  def wait(self,timeout=None):
   if self.code is not None:return self.code
   self.calls+=1
   if self.calls==1:signal.raise_signal(signal.SIGTERM)
   if self.forwarded and self.calls>=4:
    events.append('checkpoint_saved');self.code=124;events.append('worker_reaped');return self.code
   raise subprocess.TimeoutExpired('fixture',timeout)
 lock=root/'policy.lock'
 with patch.object(S.subprocess,'Popen',Fake):code=S.supervise(['fixture'],lock,30,grace_s=2)
 assert code==124 and events==['forwarded','checkpoint_saved','worker_reaped'] and not lock.exists()
 lock.write_text('other-owner',encoding='utf-8')
 try:
  with patch.object(S.subprocess,'Popen',side_effect=AssertionError('Must not spawn')):S.supervise(['fixture'],lock,30)
 except RuntimeError:pass
 else:raise AssertionError('Existing lock accepted')
 assert lock.read_text()=='other-owner';lock.unlink()
 class Hung(Fake):
  def wait(self,timeout=None):
   if self.code is not None:return self.code
   if not self.forwarded:signal.raise_signal(signal.SIGTERM)
   time.sleep(.02);raise subprocess.TimeoutExpired('hung',timeout)
 forced=[]
 with patch.object(S.subprocess,'Popen',Hung):code=S.supervise(['hung'],lock,30,grace_s=.03,on_forced_stop=forced.append)
 assert code==124 and forced==['interrupted_censored'] and not lock.exists() and 'killed' in events
 return {'parent_handler_forward_wait_cleanup':'pass','existing_lock_preserved':'pass','hung_child_kill_reap_before_cleanup':'pass'}

def fixture_worker(out,hang):
 out.mkdir(parents=True,exist_ok=True)
 def stop(*_):
  if hang:return
  time.sleep(.3)
  tmp=out/'checkpoint.saved.tmp';tmp.write_text('saved-before-parent-exit',encoding='utf-8');tmp.replace(out/'checkpoint.saved')
  raise SystemExit(124)
 signal.signal(signal.SIGTERM,stop)
 (out/'worker.ready').write_text(str(os.getpid()),encoding='utf-8')
 while True:time.sleep(.1)
def fixture_driver(out,hang):
 cmd=[sys.executable,__file__,'--fixture-worker',str(out)]
 if hang:cmd.append('--hang')
 return S.supervise(cmd,out.with_suffix('.lock'),10.,grace_s=.5 if hang else 3.,
  on_forced_stop=lambda reason:(out/'forced.reason').write_text(reason,encoding='utf-8'))

def actual_signal_check(root,group,hang=False):
 out=root/('group' if group else 'parent')
 if hang:out=root/'hung'
 driver=subprocess.Popen([sys.executable,__file__,'--fixture-driver',str(out),*(['--hang'] if hang else [])],start_new_session=True)
 try:
  deadline=time.monotonic()+10
  while not (out/'worker.ready').exists():
   if driver.poll() is not None:raise RuntimeError('Fixture ended before ready')
   if time.monotonic()>deadline:raise TimeoutError('Fixture ready timeout')
   time.sleep(.02)
  if group:os.killpg(driver.pid,signal.SIGTERM)
  else:os.kill(driver.pid,signal.SIGTERM)
  code=driver.wait(timeout=8)
  assert code==124 and not out.with_suffix('.lock').exists()
  if not hang:assert (out/'checkpoint.saved').read_text()=='saved-before-parent-exit'
  else:assert (out/'forced.reason').read_text()=='interrupted_censored'
  try:os.killpg(driver.pid,0)
  except ProcessLookupError:pass
  else:raise AssertionError('Process group still alive after parent exit')
  return 'pass'
 finally:
  if driver.poll() is None:
   os.killpg(driver.pid,signal.SIGKILL);driver.wait()

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir');p.add_argument('--fixture-worker');p.add_argument('--fixture-driver');p.add_argument('--hang',action='store_true');args=p.parse_args()
 if args.fixture_worker:fixture_worker(Path(args.fixture_worker),args.hang);return
 if args.fixture_driver:raise SystemExit(fixture_driver(Path(args.fixture_driver),args.hang))
 if not args.output_dir:p.error('--output-dir required')
 out=Path(args.output_dir).resolve()
 if out.exists() and any(out.iterdir()):raise ValueError('Use a new verification directory')
 out.mkdir(parents=True,exist_ok=True)
 result={'policy':policy_checks(out),'platform':sys.platform,'all_pass':True}
 if os.name=='posix':
  result['POSIX_signals']={'parent_only_SIGTERM':actual_signal_check(out,False),
   'process_group_SIGTERM':actual_signal_check(out,True),'hung_worker_group_SIGTERM':actual_signal_check(out,True,True)}
 else:result['POSIX_signals']='not run: Windows TerminateProcess is not POSIX SIGTERM'
 (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
