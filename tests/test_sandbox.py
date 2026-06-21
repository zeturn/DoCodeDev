from __future__ import annotations

import os
from unittest import TestCase
from unittest.mock import patch

from docode.config import load_config
from docode.sandbox import normalize_sandbox_network_mode


class SandboxPolicyTests(TestCase):
    def test_normalizes_project_and_no_internet_network_modes(self) -> None:
        self.assertEqual(normalize_sandbox_network_mode(None), "project")
        self.assertEqual(normalize_sandbox_network_mode("bridge"), "project")
        self.assertEqual(normalize_sandbox_network_mode(" internal "), "no_internet")
        self.assertEqual(normalize_sandbox_network_mode("offline"), "no_internet")

    def test_rejects_raw_docker_network_modes(self) -> None:
        for mode in ("host", "none", "container:abc", "dobox_default"):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    normalize_sandbox_network_mode(mode)

    def test_config_reads_sandbox_network_mode(self) -> None:
        with patch.dict(os.environ, {"DOCODE_SANDBOX_NETWORK_MODE": "internal"}, clear=False):
            self.assertEqual(load_config().sandbox_network_mode, "no_internet")
