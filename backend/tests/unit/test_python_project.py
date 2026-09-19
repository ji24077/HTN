"""Cancellation must stop local processes and remove the attempt's project files."""

import asyncio
import json
import os
import sys
import unittest

from orchestrator.preprocessing.artifacts import encoded
from orchestrator.shared.protocol import TaskSpec
from orchestrator.worker.executors.python_project import PythonProjectExecutor

SOURCE = """import json, os, subprocess, sys, time

def run(seed, parameters):
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    print(json.dumps({'root': os.getcwd(), 'pid': os.getpid(), 'child': child.pid}), flush=True)
    if parameters.get('wait'):
        time.sleep(60)
    return seed
"""


def spec(wait=True):
    return TaskSpec(
        id="cleanup-test",
        job_id="cleanup-job",
        kind="python_project",
        payload={"mode": "map", "module": "simulation", "seeds": [1], "parameters": {"wait": wait}},
        requirements={"runtime": "cpu", "vram_mib": 0},
        max_attempts=1,
        timeout_seconds=10,
    )


class Reporter:
    def __init__(self):
        self.events = []
        self.text = ""
        self.ready = asyncio.Event()

    def __call__(self, _):
        pass

    def emit(self, kind, data):
        self.events.append((kind, data))

    def stdout(self, text):
        self.text += text
        if "\n" in self.text:
            self.ready.set()


async def dead(pid):
    proc = await asyncio.create_subprocess_exec(
        "ps", "-o", "stat=", "-p", str(pid), stdout=asyncio.subprocess.PIPE
    )
    output, _ = await proc.communicate()
    return proc.returncode != 0 or output.strip().startswith(b"Z")


@unittest.skipIf(os.name == "nt", "POSIX local-worker process groups")
class PythonProjectTests(unittest.IsolatedAsyncioTestCase):
    async def run_project(self, wait=True):
        async def fetch(_):
            return {"simulation.py": encoded(SOURCE)}

        reporter = Reporter()
        task = asyncio.create_task(
            PythonProjectExecutor("ws://localhost", "worker", fetch=fetch).execute(
                spec(wait), reporter
            )
        )
        await asyncio.wait_for(reporter.ready.wait(), 5)
        info = json.loads(reporter.text.splitlines()[0])
        return task, reporter, info

    async def assert_vacated(self, info):
        self.assertFalse(os.path.exists(info["root"]))
        for pid in (info["pid"], info["child"]):
            for _ in range(20):
                if await dead(pid):
                    break
                await asyncio.sleep(0.05)
            self.assertTrue(await dead(pid), f"Process {pid} survived cleanup")

    async def test_cancel_kills_descendants_and_removes_inputs(self):
        task, reporter, info = await self.run_project()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        await self.assert_vacated(info)
        self.assertIn(
            ("cleaned", {"workspace_removed": True, "processes_stopped": True}), reporter.events
        )

    async def test_normal_exit_does_not_hang_on_descendant_output_pipes(self):
        task, reporter, info = await self.run_project(wait=False)
        result = await asyncio.wait_for(task, 3)
        self.assertTrue(result["ok"])
        await self.assert_vacated(info)
        self.assertTrue(any(kind == "cleaned" for kind, _ in reporter.events))

    async def test_worker_hard_exit_cleans_workspace_and_process_group(self):
        helper = """import asyncio, json
from orchestrator.preprocessing.artifacts import encoded
from orchestrator.shared.protocol import TaskSpec
from orchestrator.worker.executors.python_project import PythonProjectExecutor
source, task = json.loads(input())
async def fetch(_): return {'simulation.py': encoded(source)}
class Reporter:
    def __call__(self, _): pass
    def stdout(self, text): print(text, end='', flush=True)
asyncio.run(PythonProjectExecutor('ws://localhost', 'worker', fetch=fetch).execute(TaskSpec.model_validate(task), Reporter()))
"""
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            helper,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            process.stdin.write(
                (json.dumps([SOURCE, spec().model_dump(mode="json")]) + "\n").encode()
            )
            await process.stdin.drain()
            info = json.loads(await asyncio.wait_for(process.stdout.readline(), 5))
            process.kill()
            await asyncio.wait_for(process.wait(), 5)
            for _ in range(40):
                if not os.path.exists(info["root"]) and await dead(info["pid"]):
                    break
                await asyncio.sleep(0.05)
            await self.assert_vacated(info)
        finally:
            if process.returncode is None:
                process.kill()
            await process.wait()
