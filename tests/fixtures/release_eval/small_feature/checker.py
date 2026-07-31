from docode.eval.checker import (
    Check,
    CheckResult,
    CheckerContext,
    check,
    assert_artifact_present,
    assert_implementation_modified,
    assert_required_commands,
    assert_terminal,
    assert_tests_unmodified,
)

HIDDEN = (
    "python -c \"import slugify as m; "
    "assert m.slugify('Hello, World!')=='hello-world'; "
    "assert m.slugify('Foo & Bar')=='foo-bar'; "
    "assert m.slugify('  Multiple   Spaces  ').strip()=='multiple-spaces'; "
    "assert m.slugify('Already-Slug')=='already-slug'; "
    "print('HIDDEN_OK')\""
)


async def run_check(ctx: CheckerContext) -> CheckResult:
    checks = list(await assert_required_commands(ctx))
    checks.append(await assert_tests_unmodified(ctx, ["tests/test_slugify.py"]))
    checks.append(await assert_implementation_modified(ctx, "slugify.py"))
    code, out = await ctx.inspector.run_command(HIDDEN)
    checks.append(check("hidden_behavior", code == 0, "HIDDEN_OK" if code == 0 else out[:400]))
    checks.append(await assert_terminal(ctx))
    checks.append(await assert_artifact_present(ctx))
    passed = all(c.passed for c in checks)
    summary = "passed" if passed else "; ".join(c.details for c in checks if not c.passed)
    return CheckResult(passed=passed, checks=checks, summary=summary)
