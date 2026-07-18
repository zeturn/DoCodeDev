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
    "python -c \"import parser; "
    "assert parser.parse_pairs('a = 1')['a']=='1'; "
    "assert parser.parse_pairs('x=1\\nbadline\\ny=2')=={'x':'1','y':'2'}; "
    "assert parser.parse_pairs('  k = v  ')=={'k':'v'}; "
    "print('HIDDEN_OK')\""
)


async def run_check(ctx: CheckerContext) -> CheckResult:
    checks = list(await assert_required_commands(ctx))
    checks.append(await assert_tests_unmodified(ctx, ["tests/test_parser.py"]))
    checks.append(await assert_implementation_modified(ctx, "parser.py"))
    code, out = await ctx.inspector.run_command(HIDDEN)
    checks.append(check("hidden_behavior", code == 0, "HIDDEN_OK" if code == 0 else out[:400]))
    checks.append(await assert_terminal(ctx))
    checks.append(await assert_artifact_present(ctx))
    passed = all(c.passed for c in checks)
    summary = "passed" if passed else "; ".join(c.details for c in checks if not c.passed)
    return CheckResult(passed=passed, checks=checks, summary=summary)
