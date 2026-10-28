"""Offline tests: no AWS calls, real solver, systemd mutation, or shutdown."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import venv
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

NOTIFY = module("configure_stop_email")
CLOUD = module("run_lifetime_cloud")

class CloudControls(unittest.TestCase):
    def test_notification_scope_and_no_role(self):
        plan = NOTIFY.build_plan("123456789012", "us-east-1", "i-1234567890abcdef0", "example@example.com")
        self.assertEqual(plan["event_pattern"]["detail"]["instance-id"], ["i-1234567890abcdef0"])
        self.assertEqual(plan["event_pattern"]["detail"]["state"], ["stopped", "terminated"])
        self.assertEqual(plan["publish_statement"]["Condition"]["ArnEquals"]["aws:SourceArn"], plan["rule_arn"])
        self.assertEqual(plan["publish_statement"]["Resource"], plan["topic_arn"])
        self.assertNotIn("RoleArn", plan["target"])
        self.assertIn("실험 성공을 뜻하지", json.loads(plan["target"]["InputTransformer"]["InputTemplate"]))

    def test_notification_rejects_wrong_ids(self):
        with self.assertRaises(ValueError):
            NOTIFY.build_plan("1234", "us-east-1", "i-1234567890abcdef0", "example@example.com")
        with self.assertRaises(ValueError):
            NOTIFY.build_plan("123456789012", "us-east-1", "i-1234567890abcdef0", "bad address")

    def test_pending_subscription_arn_is_not_confirmation(self):
        sub = {"SubscriptionArn":"arn:aws:sns:us-east-1:123456789012:topic:pending-id"}
        with patch.object(NOTIFY, "aws", return_value={"Attributes":{"PendingConfirmation":"true"}}):
            self.assertFalse(NOTIFY.inspect_subscription("us-east-1", sub)["confirmed"])
        with patch.object(NOTIFY, "aws", return_value={"Attributes":{"PendingConfirmation":"false"}}):
            self.assertTrue(NOTIFY.inspect_subscription("us-east-1", sub)["confirmed"])
        with patch.object(NOTIFY, "aws", return_value={"Attributes":{}}):
            self.assertFalse(NOTIFY.inspect_subscription("us-east-1", sub)["confirmed"])

    def test_compact_outcome_does_not_export_history(self):
        huge = {"status":"max_efc_censored", "ledger":{"efc":3000},
            "cycles":[{"large":"x"*1000}]*10000, "checks":[1]*10000,
            "diagnostics":[{"efc":2990,"SOH_percent":95.0}]*302,
            "files":["file"]*10000, "elapsed_wall_s":70000,
            "eol":None}
        result = CLOUD.compact_outcome(huge, None, Path("results"))
        self.assertLess(len(json.dumps(result)), 1000)
        self.assertEqual(result["last_valid_EFC"], 3000)
        self.assertEqual(result["last_SOH_percent"], 95.0)
        self.assertNotIn("cycles", result)

    def test_terminate_waits_for_worker_after_parent_exit(self):
        proc=SimpleNamespace(pid=1234,poll=lambda:0,wait=lambda **kwargs:0)
        fake_os=SimpleNamespace(killpg=lambda *args:None)
        fake_signal=SimpleNamespace(SIGTERM=15,SIGKILL=9)
        with patch.object(CLOUD,"os",fake_os), patch.object(CLOUD,"signal",fake_signal), \
             patch.object(CLOUD,"process_group_alive",side_effect=[True,True,False,False,False]), \
             patch.object(CLOUD.time,"sleep") as sleeper:
            result=CLOUD.terminate_tree(proc)
        self.assertTrue(result["term_sent"])
        self.assertTrue(result["group_empty"])
        self.assertTrue(sleeper.called)
        self.assertFalse(result["kill_sent"])

    def test_terminate_escalates_and_waits_for_group(self):
        proc=SimpleNamespace(pid=1234,poll=lambda:0,wait=lambda **kwargs:0)
        killed=[]
        fake_os=SimpleNamespace(killpg=lambda *args:killed.append(args))
        fake_signal=SimpleNamespace(SIGTERM=15,SIGKILL=9)
        with patch.object(CLOUD,"os",fake_os), patch.object(CLOUD,"signal",fake_signal), \
             patch.object(CLOUD,"process_group_alive",side_effect=[True,True,True,False,False]), \
             patch.object(CLOUD.time,"monotonic",side_effect=[0,1,2]), \
             patch.object(CLOUD.time,"sleep"):
            result=CLOUD.terminate_tree(proc,grace_s=0)
        self.assertEqual(killed,[(1234,15),(1234,9)])
        self.assertTrue(result["kill_sent"])
        self.assertTrue(result["group_empty"])

    def settings(self, tmp):
        repo = Path(tmp) / "repo"
        (repo / "simulation").mkdir(parents=True)
        (repo / "simulation/run_lifetime.py").write_text("# mock", encoding="utf-8")
        run = Path(tmp) / "run"
        return SimpleNamespace(repo_root=str(repo), python=sys.executable,
            run_root=str(run), cell=4, current=3.2, mesh_scale=4, efc=3000,
            max_wall_s=72000, rpt_every_efc=10, resume=False,
            import_100efc_dir=str(Path(tmp) / "seed"),
            hard_stop_extra_s=900, guard_unit="capstone-lifetime-hard-stop",
            minimum_free_gib=2, minimum_available_ram_fraction=.08, plan=False)

    def test_cli_contract_and_separate_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.settings(tmp)
            command = CLOUD.runner_command(args)
            self.assertEqual(command[command.index("--efc")+1], "3000")
            self.assertEqual(command[command.index("--max-wall-s")+1], "72000")
            self.assertEqual(command[command.index("--output-dir")+1], str(Path(args.run_root) / "trajectory"))
            self.assertIn("--import-100efc-dir", command)
            args.resume = True
            self.assertIn("--resume", CLOUD.runner_command(args))
            self.assertNotIn("--import-100efc-dir", CLOUD.runner_command(args))
            self.assertIn("--on-active=72900s", CLOUD.hard_stop_command(args))

    @unittest.skipUnless(sys.platform.startswith("linux"), "real venv symlink fixture requires Linux")
    def test_venv_symlink_preserves_prefix_and_venv_only_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.settings(tmp)
            prefix = Path(tmp) / "fixture-venv"
            venv.EnvBuilder(with_pip=False, symlinks=True).create(prefix)
            launcher = prefix / "bin/python"
            self.assertTrue(launcher.is_symlink())
            args.python = str(launcher)
            command = CLOUD.runner_command(args)
            self.assertEqual(command[0], str(launcher.absolute()))
            purelib = subprocess.run([command[0], "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))"],
                check=True, capture_output=True, text=True).stdout.strip()
            module_name = "capstone_venv_only_regression_fixture"
            (Path(purelib) / (module_name + ".py")).write_text("VALUE = 'venv-only'\n", encoding="utf-8")
            probe = subprocess.run([command[0], "-c",
                "import json,sys," + module_name + "; print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,'value':" + module_name + ".VALUE}))"],
                check=True, capture_output=True, text=True)
            value = json.loads(probe.stdout)
            self.assertEqual(Path(value["prefix"]).resolve(), prefix.resolve())
            self.assertNotEqual(value["prefix"], value["base_prefix"])
            self.assertEqual(value["value"], "venv-only")
            # The old resolve() command would choose this base interpreter.
            bad_probe = subprocess.run([str(launcher.resolve()), "-c",
                "import importlib.util; print(importlib.util.find_spec('" + module_name + "') is None)"],
                check=True, capture_output=True, text=True)
            self.assertEqual(bad_probe.stdout.strip(), "True")

    def test_result_preserved_then_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.settings(tmp)
            child = Path(args.run_root) / "trajectory"
            child.mkdir(parents=True)
            (child / "outcome.json").write_text('{"status":"eol_observed"}', encoding="utf-8")
            calls = []
            saved = []
            fake_os = SimpleNamespace(name="posix", geteuid=lambda: 0,
                environ={}, fsync=CLOUD.os.fsync)
            proc = SimpleNamespace(pid=1234, poll=lambda: 0, wait=lambda **kwargs: 0)
            def call(cmd, **kwargs):
                calls.append(cmd)
                return SimpleNamespace(returncode=0)
            with patch.object(CLOUD, "os", fake_os), patch.object(CLOUD.subprocess, "run", side_effect=call), \
                 patch.object(CLOUD.subprocess, "Popen", return_value=proc), \
                 patch.object(CLOUD.shutil, "disk_usage", return_value=SimpleNamespace(free=10*CLOUD.GIB)), \
                 patch.object(CLOUD, "terminate_tree", return_value={"term_sent":False,"kill_sent":False,"group_empty":True}), \
                 patch.object(CLOUD, "save_json", side_effect=lambda path,value: saved.append(copy.deepcopy(value))):
                self.assertEqual(CLOUD.run(args), 0)
            final = saved[-1]
            self.assertEqual(final["status"], "eol_observed")
            self.assertTrue(final["shutdown_requested"])
            self.assertEqual(calls[0][0], "systemd-run")
            self.assertEqual(calls[-2], ["sync"])
            self.assertEqual(calls[-1], ["/usr/sbin/shutdown", "-h", "now"])

    def test_runner_launch_failure_still_records_and_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.settings(tmp)
            calls = []
            saved = []
            fake_os = SimpleNamespace(name="posix", geteuid=lambda: 0,
                environ={}, fsync=CLOUD.os.fsync)
            def call(cmd, **kwargs):
                calls.append(cmd)
                return SimpleNamespace(returncode=0)
            with patch.object(CLOUD, "os", fake_os), patch.object(CLOUD.subprocess, "run", side_effect=call), \
                 patch.object(CLOUD.subprocess, "Popen", side_effect=RuntimeError("mock launch failure")), \
                 patch.object(CLOUD.shutil, "disk_usage", return_value=SimpleNamespace(free=10*CLOUD.GIB)), \
                 patch.object(CLOUD, "terminate_tree", return_value={"term_sent":False,"kill_sent":False,"group_empty":True}), \
                 patch.object(CLOUD, "save_json", side_effect=lambda path,value: saved.append(copy.deepcopy(value))):
                self.assertEqual(CLOUD.run(args), 1)
            final = saved[-1]
            self.assertEqual(final["status"], "supervisor_failed")
            self.assertEqual(calls[-1], ["/usr/sbin/shutdown", "-h", "now"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
