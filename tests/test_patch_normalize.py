from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest

from docode.dobox.patch_normalize import (
    convert_aider_patch,
    normalize_model_patch,
    strip_code_fences,
)

REAL_UPDATE = (
    "*** Begin Patch\n"
    "*** Update File: /workspace/calc.go\n"
    "@@\n"
    " func Add(a, b int) int {\n"
    "-\treturn a - b // BUG: should return a + b\n"
    "+\treturn a + b\n"
    " }\n"
    "*** End Patch"
)

REAL_ADD = (
    "*** Begin Patch\n"
    "*** Add File: /workspace/new.py\n"
    "@@\n"
    "print('hello')\n"
    "*** End Patch"
)


class PatchNormalizeTests(unittest.TestCase):
    def test_strip_code_fences_full(self) -> None:
        self.assertEqual(strip_code_fences("```diff\nfoo\n```"), "foo")
        self.assertEqual(strip_code_fences("```\nfoo\n```"), "foo")

    def test_strip_code_fences_leading_only(self) -> None:
        self.assertEqual(strip_code_fences("```diff\nfoo"), "foo")

    def test_strip_code_fences_none(self) -> None:
        self.assertEqual(strip_code_fences("plain diff"), "plain diff")

    def test_convert_aider_update_strips_workspace_and_markers(self) -> None:
        out = convert_aider_patch(REAL_UPDATE)
        self.assertIn("diff --git a/calc.go b/calc.go", out)
        self.assertIn("--- a/calc.go", out)
        self.assertIn("+++ b/calc.go", out)
        self.assertIn("@@ -1,3 +1,3 @@", out)
        self.assertIn("-\treturn a - b // BUG: should return a + b", out)
        self.assertIn("+\treturn a + b", out)
        self.assertNotIn("*** Begin Patch", out)
        self.assertNotIn("\n@@\n", out)
        self.assertNotIn("/workspace/calc.go", out)

    def test_convert_aider_add_prefixes_plus(self) -> None:
        out = convert_aider_patch(REAL_ADD)
        self.assertIn("--- /dev/null", out)
        self.assertIn("+++ b/new.py", out)
        self.assertIn("+print('hello')", out)

    def test_normalize_model_patch_fenced_envelope(self) -> None:
        out = normalize_model_patch("```diff\n" + REAL_UPDATE + "\n```")
        self.assertIn("diff --git a/calc.go b/calc.go", out)

    def test_integration_git_apply_unidiff_zero(self) -> None:
        tmp = tempfile.mkdtemp(prefix="patch-norm-")
        try:
            repo = os.path.join(tmp, "repo")
            os.makedirs(repo)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
            original = (
                "package main\n\n"
                "func Add(a, b int) int {\n"
                "\treturn a - b // BUG: should return a + b\n"
                "}\n"
            )
            with open(os.path.join(repo, "calc.go"), "w", encoding="utf-8") as fh:
                fh.write(original)
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)

            normalized = normalize_model_patch(REAL_UPDATE)
            diff_path = os.path.join(tmp, "p.diff")
            with open(diff_path, "w", encoding="utf-8") as fh:
                fh.write(normalized)
            result = subprocess.run(
                ["git", "apply", "--unidiff-zero", "--ignore-whitespace", diff_path],
                cwd=repo,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            with open(os.path.join(repo, "calc.go"), "r", encoding="utf-8") as fh:
                applied = fh.read()
            self.assertIn("\treturn a + b", applied)
            self.assertNotIn("// BUG", applied)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
