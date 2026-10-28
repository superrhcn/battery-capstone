"""Bounded parent/worker shutdown with owned-lock cleanup (no physics imports)."""
from __future__ import annotations
import os
from pathlib import Path
import signal
import subprocess
import time


def supervise(command,lock_path,wall_remaining_s,*,grace_s=60.,on_forced_stop=None,
              external_stop_event=None):
 """Return worker exit code only after it has exited.

 SIGTERM/SIGINT set a request flag, forwarded once to the worker. The parent
 remains alive while the worker finalizes its checkpoint. Repeated signals do
 not restart the grace clock. A hung worker is killed and reaped before cleanup.
 external_stop_event is dependency injection for deterministic local tests.
 """
 lock_path=Path(lock_path);owned=False;proc=None
 requested={'signal':None};previous={};forwarded_at=None
 began=time.monotonic();hard_deadline=began+wall_remaining_s+15.
 def request_stop(signum,_frame):
  if requested['signal'] is None:requested['signal']=signum
 for sig in (signal.SIGTERM,signal.SIGINT):
  previous[sig]=signal.getsignal(sig);signal.signal(sig,request_stop)
 def stop_and_reap():
  if proc is None or proc.poll() is not None:return
  proc.terminate()
  try:proc.wait(timeout=grace_s)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
 try:
  try:fd=os.open(lock_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
  except FileExistsError:raise RuntimeError('Run lock exists; inspect live process before removing stale lock')
  owned=True
  with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
  # A stop arriving before spawn must not leave a live worker behind.
  if requested['signal'] is not None:return 124
  proc=subprocess.Popen(command)
  while True:
   completed=proc.poll()
   if completed is not None:return completed
   now=time.monotonic()
   if external_stop_event is not None and external_stop_event.is_set() and requested['signal'] is None:
    requested['signal']=signal.SIGTERM
   if requested['signal'] is not None and forwarded_at is None:
    forwarded_at=now
    if proc.poll() is None:proc.send_signal(signal.SIGTERM)
   force_reason=None
   if now>=hard_deadline:force_reason='wall_time_censored'
   elif forwarded_at is not None and now-forwarded_at>=grace_s:force_reason='interrupted_censored'
   if force_reason:
    completed=proc.poll()
    if completed is not None:return completed
    proc.kill()
    completed=proc.wait()
    if completed==0:return 0
    if on_forced_stop:on_forced_stop(force_reason)
    return 124
   wait_for=min(1.,max(.01,hard_deadline-now))
   if forwarded_at is not None:wait_for=min(wait_for,max(.01,grace_s-(now-forwarded_at)))
   try:return proc.wait(timeout=wait_for)
   except subprocess.TimeoutExpired:pass
 finally:
  # Exception or cancellation cannot release the lock while a child is alive.
  try:stop_and_reap()
  finally:
   if owned:lock_path.unlink(missing_ok=True)
   for sig,handler in previous.items():signal.signal(sig,handler)
