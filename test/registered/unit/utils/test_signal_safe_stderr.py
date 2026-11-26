"""Unit tests for signal-safe stderr writes in srt/utils/process.py."""

import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from sglang.srt.utils import process as process_module
from sglang.srt.utils.process import kill_process_tree, write_stderr_signal_safe
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

_REENTRANT = "reentrant call inside <_io.BufferedWriter name='<stderr>'>"


class _ReentrantStderr:
    def write(self, _):
        raise RuntimeError(_REENTRANT)

    def flush(self):
        raise RuntimeError(_REENTRANT)


class TestWriteStderrSignalSafe(CustomTestCase):
    def test_writes_to_fd2_without_sys_stderr(self):
        with tempfile.TemporaryFile() as captured:
            saved_fd2 = os.dup(2)
            try:
                os.dup2(captured.fileno(), 2)
                with mock.patch.object(sys, "stderr", _ReentrantStderr()):
                    write_stderr_signal_safe("child failed")
            finally:
                os.dup2(saved_fd2, 2)
                os.close(saved_fd2)
            captured.seek(0)
            self.assertEqual(captured.read(), b"child failed\n")

    def test_retries_short_writes(self):
        with tempfile.TemporaryFile() as captured:
            saved_fd2 = os.dup(2)
            real_write = os.write
            written = []

            def short_write(fd, data):
                chunk = data[:3]
                written.append(chunk)
                return real_write(fd, chunk)

            try:
                os.dup2(captured.fileno(), 2)
                with mock.patch.object(os, "write", side_effect=short_write):
                    write_stderr_signal_safe("abcdef")
            finally:
                os.dup2(saved_fd2, 2)
                os.close(saved_fd2)
            captured.seek(0)
            self.assertEqual(captured.read(), b"abcdef\n")
            self.assertGreater(len(written), 2)


class TestKillProcessTreeWaitWhenLoggingReenters(CustomTestCase):
    def test_wait_warning_does_not_abort_reaping(self):
        proc = mock.Mock(pid=1234)
        with (
            mock.patch.object(
                process_module,
                "_still_holding_resources",
                side_effect=[[proc], []],
            ),
            mock.patch.object(
                process_module.time, "monotonic", side_effect=[0.0, 0.0, 11.0]
            ),
            mock.patch.object(process_module.time, "sleep"),
            mock.patch.object(
                process_module.logger,
                "warning",
                side_effect=RuntimeError(_REENTRANT),
            ),
            mock.patch.object(process_module, "write_stderr_signal_safe") as fallback,
        ):
            process_module._wait_for_reap_or_raise([proc], wait_timeout=60)

        fallback.assert_called_once()
        self.assertIn("still alive", fallback.call_args.args[0])


class TestSigquitHandlerCleanup(CustomTestCase):
    def test_kills_when_crash_dump_raises(self):
        from sglang.srt.managers.tokenizer_manager import SignalHandler

        tokenizer_manager = mock.Mock()
        tokenizer_manager._subprocess_watchdog = None
        tokenizer_manager.dump_requests_before_crash.side_effect = RuntimeError(
            _REENTRANT
        )
        handler = SignalHandler(tokenizer_manager)

        with (
            mock.patch(
                "sglang.srt.managers.tokenizer_manager.kill_process_tree"
            ) as kill,
            mock.patch(
                "sglang.srt.managers.tokenizer_manager.write_stderr_signal_safe"
            ) as write,
        ):
            handler.running_phase_sigquit_handler()

        kill.assert_called_once_with(os.getpid())
        self.assertTrue(
            any("cleanup failed" in str(call.args[0]) for call in write.call_args_list)
        )


class TestKillProcessTreeWhenLoggingReenters(CustomTestCase):
    def test_kills_even_if_logging_raises_reentrant_error(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            with mock.patch(
                "sglang.srt.utils.process.logger.info",
                side_effect=RuntimeError(_REENTRANT),
            ):
                kill_process_tree(child.pid, include_parent=True, wait_timeout=30)
            self.assertIsNotNone(child.wait(timeout=10))
        finally:
            if child.poll() is None:
                child.kill()


if __name__ == "__main__":
    unittest.main()
