"""Phase 1 tests: data-contract stability for StepOutcome & FinalizationBlocker."""

from __future__ import annotations

import json
from unittest import TestCase

from docode.agent.outcome import (
    BlockerSource,
    FinalizationBlocker,
    OutcomeKind,
    RequiredAction,
    StepOutcome,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _make_blocker(**overrides) -> FinalizationBlocker:
    kwargs = {
        "code": "verification_stale",
        "source": BlockerSource.VERIFICATION_SCHEDULER,
        "message": "Run the required command.",
        "required_action": RequiredAction.RUN_REQUIRED_COMMAND,
        "related_files": ("src/a.py",),
        "related_commands": ("python -m unittest tests.test_a",),
        "related_node_ids": ("verify",),
        "evidence_refs": ("command:1:test",),
        "retryable": True,
    }
    kwargs.update(overrides)
    return FinalizationBlocker(**kwargs)


def _make_outcome(**overrides) -> StepOutcome:
    kwargs: dict = {
        "kind": OutcomeKind.TOOL,
        "action_key": "run_command:python -m pytest",
        "success": True,
        "progress": True,
        "progress_reasons": ("fresh evidence",),
        "state_fingerprint_before": "aaa",
        "state_fingerprint_after": "bbb",
        "workspace_changed": False,
        "evidence_added": ("command:1:pytest",),
        "blockers": (),
        "next_required_action": RequiredAction.NONE,
    }
    kwargs.update(overrides)
    return StepOutcome(**kwargs)


# ── 11.1  enum value stability ──────────────────────────────────────────


class EnumStabilityTests(TestCase):
    def test_required_action_values(self) -> None:
        self.assertEqual(RequiredAction.NONE.value, "none")
        self.assertEqual(RequiredAction.INSPECT_TARGET.value, "inspect_target")
        self.assertEqual(RequiredAction.EDIT_TARGET.value, "edit_target")
        self.assertEqual(RequiredAction.RUN_REQUIRED_COMMAND.value, "run_required_command")
        self.assertEqual(RequiredAction.STOP_NON_CONVERGENT.value, "stop_non_convergent")

    def test_blocker_source_values(self) -> None:
        self.assertEqual(BlockerSource.WORKFLOW.value, "workflow")
        self.assertEqual(BlockerSource.NO_PROGRESS.value, "no_progress")
        self.assertEqual(BlockerSource.MODEL.value, "model")

    def test_outcome_kind_values(self) -> None:
        self.assertEqual(OutcomeKind.TOOL.value, "tool")
        self.assertEqual(OutcomeKind.DECISION_REJECTED.value, "decision_rejected")
        self.assertEqual(OutcomeKind.EXPORT.value, "export")


# ── 11.2  blocker JSON serialisation ────────────────────────────────────


class BlockerSerialisationTests(TestCase):
    def test_full_blocker_json_roundtrip(self) -> None:
        blocker = _make_blocker()
        d = blocker.to_dict()
        raw = json.dumps(d)
        parsed = json.loads(raw)

        self.assertEqual(parsed["code"], "verification_stale")
        self.assertEqual(parsed["source"], "verification_scheduler")
        self.assertIsInstance(parsed["source"], str)
        self.assertEqual(parsed["required_action"], "run_required_command")
        self.assertIsInstance(parsed["related_files"], list)
        self.assertEqual(parsed["related_files"], ["src/a.py"])
        self.assertEqual(parsed["related_commands"], ["python -m unittest tests.test_a"])
        self.assertEqual(parsed["related_node_ids"], ["verify"])
        self.assertEqual(parsed["evidence_refs"], ["command:1:test"])
        self.assertTrue(parsed["retryable"])
        self.assertEqual(len(parsed["fingerprint"]), 64)
        self.assertEqual(parsed["fingerprint"], blocker.fingerprint())


# ── 11.3  input order does not affect blocker fingerprint ────────────────


class BlockerOrderTests(TestCase):
    def test_file_order_independence(self) -> None:
        a = _make_blocker(related_files=("b.py", "a.py"))
        b = _make_blocker(related_files=("a.py", "b.py"))
        self.assertEqual(a.fingerprint(), b.fingerprint())
        self.assertEqual(a.related_files, ("a.py", "b.py"))

    def test_command_order_independence(self) -> None:
        a = _make_blocker(related_commands=("b", "a"))
        b = _make_blocker(related_commands=("a", "b"))
        self.assertEqual(a.fingerprint(), b.fingerprint())

    def test_node_order_independence(self) -> None:
        a = _make_blocker(related_node_ids=("y", "x"))
        b = _make_blocker(related_node_ids=("x", "y"))
        self.assertEqual(a.fingerprint(), b.fingerprint())


# ── 11.4  duplicate values do not affect blocker fingerprint ─────────────


class BlockerDedupTests(TestCase):
    def test_duplicate_related_files(self) -> None:
        a = _make_blocker(related_files=("a.py", "a.py", "b.py"))
        b = _make_blocker(related_files=("a.py", "b.py"))
        self.assertEqual(a.fingerprint(), b.fingerprint())
        self.assertEqual(a.related_files, ("a.py", "b.py"))

    def test_duplicate_commands(self) -> None:
        a = _make_blocker(related_commands=("x", "x"))
        b = _make_blocker(related_commands=("x",))
        self.assertEqual(a.fingerprint(), b.fingerprint())


# ── 11.5  message does not affect blocker fingerprint ────────────────────


class BlockerMessageTests(TestCase):
    def test_different_message_same_fingerprint(self) -> None:
        a = _make_blocker(message="Run pytest.")
        b = _make_blocker(message="The required test is stale. Run pytest now.")
        self.assertEqual(a.fingerprint(), b.fingerprint())

    def test_message_still_appears_in_dict(self) -> None:
        a = _make_blocker(message="custom")
        d = a.to_dict()
        self.assertEqual(d["message"], "custom")


# ── 11.6  evidence_refs does not affect blocker fingerprint ──────────────


class BlockerEvidenceTests(TestCase):
    def test_different_evidence_same_fingerprint(self) -> None:
        a = _make_blocker(evidence_refs=("cmd:1",))
        b = _make_blocker(evidence_refs=("cmd:2", "cmd:3"))
        self.assertEqual(a.fingerprint(), b.fingerprint())

    def test_evidence_preserved_in_dict(self) -> None:
        a = _make_blocker(evidence_refs=("a", "b"))
        d = a.to_dict()
        self.assertEqual(d["evidence_refs"], ["a", "b"])


# ── 11.7  retryable affects fingerprint ─────────────────────────────────


class BlockerRetryableTests(TestCase):
    def test_retryable_changes_fingerprint(self) -> None:
        a = _make_blocker(retryable=True)
        b = _make_blocker(retryable=False)
        self.assertNotEqual(a.fingerprint(), b.fingerprint())


# ── 11.8  path slash normalisation ──────────────────────────────────────


class BlockerSlashTests(TestCase):
    def test_backslash_normalised_to_slash(self) -> None:
        blocker = _make_blocker(related_files=("src\\docode\\a.py",))
        self.assertEqual(blocker.related_files, ("src/docode/a.py",))

    def test_mixed_slashes(self) -> None:
        blocker = _make_blocker(related_files=("a\\b.py", "c/d.py"))
        self.assertEqual(blocker.related_files, ("a/b.py", "c/d.py"))

    def test_slash_normalisation_deduplicates_collisions(self) -> None:
        blocker = _make_blocker(
            related_files=(
                "src\\docode\\a.py",
                "src/docode/a.py",
            )
        )
        self.assertEqual(
            blocker.related_files,
            ("src/docode/a.py",),
        )

    def test_semantically_identical_paths_have_same_fingerprint(self) -> None:
        windows = _make_blocker(related_files=("src\\docode\\a.py",))
        posix = _make_blocker(related_files=("src/docode/a.py",))
        self.assertEqual(windows.fingerprint(), posix.fingerprint())


# ── 11.9  empty blocker code rejected ────────────────────────────────────


class BlockerValidationTests(TestCase):
    def test_empty_code_raises(self) -> None:
        with self.assertRaises(ValueError):
            _make_blocker(code="")
        with self.assertRaises(ValueError):
            _make_blocker(code="   ")

    def test_empty_message_falls_back_to_code(self) -> None:
        blocker = _make_blocker(message="")
        self.assertEqual(blocker.message, blocker.code)


# ── 11.10  StepOutcome JSON serialisation ────────────────────────────────


class OutcomeSerialisationTests(TestCase):
    def test_full_outcome_json_roundtrip(self) -> None:
        blocker = _make_blocker()
        outcome = _make_outcome(
            blockers=(blocker,),
            success=True,
            progress=True,
        )
        d = outcome.to_dict()
        raw = json.dumps(d)
        parsed = json.loads(raw)

        self.assertEqual(parsed["kind"], "tool")
        self.assertEqual(parsed["success"], True)
        self.assertEqual(parsed["progress"], True)
        self.assertEqual(len(parsed["blockers"]), 1)
        self.assertEqual(len(parsed["fingerprint"]), 64)
        self.assertIsInstance(parsed["primary_blocker"], dict)


# ── 11.11  success and progress are independent ──────────────────────────


class SuccessProgressIndependenceTests(TestCase):
    def test_success_true_progress_false_not_overridden(self) -> None:
        outcome = _make_outcome(success=True, progress=False)
        self.assertTrue(outcome.success)
        self.assertFalse(outcome.progress)

    def test_success_false_progress_true_not_overridden(self) -> None:
        outcome = _make_outcome(success=False, progress=True)
        self.assertFalse(outcome.success)
        self.assertTrue(outcome.progress)


# ── 11.12  blocker order does not affect outcome fingerprint ─────────────


class OutcomeBlockerOrderTests(TestCase):
    def test_reversed_blocker_order_same_fingerprint(self) -> None:
        a = _make_blocker(code="a")
        b = _make_blocker(code="b")
        o1 = _make_outcome(blockers=(a, b))
        o2 = _make_outcome(blockers=(b, a))
        self.assertEqual(o1.fingerprint(), o2.fingerprint())
        self.assertEqual(
            o1.primary_blocker().code,
            o2.primary_blocker().code,
        )
        self.assertEqual(
            [bl["code"] for bl in o1.to_dict()["blockers"]],
            [bl["code"] for bl in o2.to_dict()["blockers"]],
        )


# ── 11.13  duplicate blocker deduplication ──────────────────────────────


class BlockerDedupInOutcomeTests(TestCase):
    def test_semantically_identical_blockers_dedup(self) -> None:
        a = _make_blocker(code="dup", message="first")
        b = _make_blocker(code="dup", message="second", evidence_refs=("X",))
        # Same semantic fingerprint (message + evidence_refs excluded)
        self.assertEqual(a.fingerprint(), b.fingerprint())

        outcome = _make_outcome(blockers=(a, b))
        self.assertEqual(len(outcome.blockers), 1)
        # Keeps the first input (a)
        self.assertEqual(outcome.blockers[0].message, "first")


# ── 11.14  primary blocker priority ─────────────────────────────────────


class PrimaryBlockerPriorityTests(TestCase):
    def test_no_progress_has_highest_priority(self) -> None:
        final = _make_blocker(
            code="diff_empty",
            source=BlockerSource.FINALIZATION,
            required_action=RequiredAction.EDIT_TARGET,
        )
        sched = _make_blocker(
            code="verification_stale",
            source=BlockerSource.VERIFICATION_SCHEDULER,
            required_action=RequiredAction.RUN_REQUIRED_COMMAND,
        )
        np = _make_blocker(
            code="repeated_action_blocked",
            source=BlockerSource.NO_PROGRESS,
            required_action=RequiredAction.CHOOSE_DIFFERENT_ACTION,
        )
        outcome = _make_outcome(blockers=(final, sched, np))
        self.assertEqual(outcome.primary_blocker().source, BlockerSource.NO_PROGRESS)

        outcome2 = _make_outcome(blockers=(final, sched))
        self.assertEqual(
            outcome2.primary_blocker().source,
            BlockerSource.VERIFICATION_SCHEDULER,
        )


# ── 11.15  effective required action ────────────────────────────────────


class EffectiveRequiredActionTests(TestCase):
    def test_explicit_action_has_priority(self) -> None:
        blocker = _make_blocker(
            required_action=RequiredAction.RUN_REQUIRED_COMMAND,
        )
        outcome = _make_outcome(
            blockers=(blocker,),
            next_required_action=RequiredAction.CHOOSE_DIFFERENT_ACTION,
        )
        self.assertEqual(
            outcome.effective_required_action(),
            RequiredAction.CHOOSE_DIFFERENT_ACTION,
        )

    def test_falls_back_to_blocker(self) -> None:
        outcome = _make_outcome(
            blockers=(_make_blocker(required_action=RequiredAction.EDIT_TARGET),),
            next_required_action=RequiredAction.NONE,
        )
        self.assertEqual(
            outcome.effective_required_action(),
            RequiredAction.EDIT_TARGET,
        )

    def test_returns_none_when_no_blocker_no_explicit(self) -> None:
        outcome = _make_outcome(blockers=())
        self.assertEqual(
            outcome.effective_required_action(),
            RequiredAction.NONE,
        )


# ── 11.16  tuple normalisation ──────────────────────────────────────────


class TupleNormalisationTests(TestCase):
    def test_normalises_progress_reasons(self) -> None:
        outcome = _make_outcome(progress_reasons=("", "a", "a", "b"))
        self.assertEqual(outcome.progress_reasons, ("a", "b"))

    def test_normalises_evidence_added(self) -> None:
        outcome = _make_outcome(evidence_added=("b", "a", "", "a"))
        self.assertEqual(outcome.evidence_added, ("a", "b"))

    def test_normalises_completed_node_ids(self) -> None:
        outcome = _make_outcome(completed_node_ids=("", "x"))
        self.assertEqual(outcome.completed_node_ids, ("x",))

    def test_normalises_invalidated_node_ids(self) -> None:
        outcome = _make_outcome(invalidated_node_ids=("y", "x", "x"))
        self.assertEqual(outcome.invalidated_node_ids, ("x", "y"))


# ── 11.17  outcome fingerprint stable ───────────────────────────────────


class OutcomeFingerprintStabilityTests(TestCase):
    def test_tuple_order_does_not_affect_fingerprint(self) -> None:
        o1 = _make_outcome(evidence_added=("b", "a"))
        o2 = _make_outcome(evidence_added=("a", "b"))
        self.assertEqual(o1.fingerprint(), o2.fingerprint())


# ── 11.18  outcome key changes alter fingerprint ────────────────────────


class OutcomeFingerprintChangeTests(TestCase):
    def setUp(self) -> None:
        self.base = _make_outcome().fingerprint()

    def test_action_key_change(self) -> None:
        self.assertNotEqual(self.base, _make_outcome(action_key="other").fingerprint())

    def test_success_change(self) -> None:
        self.assertNotEqual(
            self.base,
            _make_outcome(success=not _make_outcome().success).fingerprint(),
        )

    def test_progress_change(self) -> None:
        self.assertNotEqual(
            self.base,
            _make_outcome(progress=not _make_outcome().progress).fingerprint(),
        )

    def test_workspace_changed(self) -> None:
        self.assertNotEqual(
            self.base,
            _make_outcome(workspace_changed=True).fingerprint(),
        )

    def test_state_fingerprint_after_change(self) -> None:
        self.assertNotEqual(
            self.base,
            _make_outcome(state_fingerprint_after="zzz").fingerprint(),
        )

    def test_blocker_change(self) -> None:
        self.assertNotEqual(
            self.base,
            _make_outcome(
                blockers=(_make_blocker(code="x"),)
            ).fingerprint(),
        )

    def test_effective_required_action_change(self) -> None:
        self.assertNotEqual(
            self.base,
            _make_outcome(
                next_required_action=RequiredAction.EDIT_TARGET,
            ).fingerprint(),
        )

    def test_failure_class_change(self) -> None:
        self.assertNotEqual(
            self.base,
            _make_outcome(failure_class="repair_non_convergent").fingerprint(),
        )


# ── 11.19  empty action_key rejected ────────────────────────────────────


class OutcomeValidationTests(TestCase):
    def test_empty_action_key_raises(self) -> None:
        with self.assertRaises(ValueError):
            _make_outcome(action_key="")
        with self.assertRaises(ValueError):
            _make_outcome(action_key="   ")


# ── 11.20  frozen / slots behaviour ─────────────────────────────────────


class FrozenSlotsTests(TestCase):
    def test_blocker_is_frozen(self) -> None:
        blocker = _make_blocker()
        with self.assertRaises(Exception):
            blocker.code = "x"  # type: ignore[misc]

    def test_outcome_is_frozen(self) -> None:
        outcome = _make_outcome()
        with self.assertRaises(Exception):
            outcome.progress = True  # type: ignore[misc]

    def test_no_arbitrary_dynamic_fields(self) -> None:
        blocker = _make_blocker()
        with self.assertRaises(AttributeError):
            _ = blocker.whiskers  # type: ignore[attr-defined]
        outcome = _make_outcome()
        with self.assertRaises(AttributeError):
            _ = outcome.whiskers  # type: ignore[attr-defined]
