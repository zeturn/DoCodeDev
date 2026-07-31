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
    "node -e \"const {add,mul}=require('./calc'); const assert=require('assert'); "
    "assert.strictEqual(add(5,5),10); assert.strictEqual(add(-1,1),0); "
    "assert.strictEqual(mul(7,6),42); console.log('HIDDEN_OK')\""
)


async def run_check(ctx: CheckerContext) -> CheckResult:
    checks = list(await assert_required_commands(ctx))
    checks.append(await assert_tests_unmodified(ctx, ["tests/calc.test.js"]))
    checks.append(await assert_implementation_modified(ctx, "calc.js"))
    code, out = await ctx.inspector.run_command(HIDDEN)
    checks.append(check("hidden_behavior", code == 0, "HIDDEN_OK" if code == 0 else out[:400]))
    checks.append(await assert_terminal(ctx))
    checks.append(await assert_artifact_present(ctx))
    passed = all(c.passed for c in checks)
    summary = "passed" if passed else "; ".join(c.details for c in checks if not c.passed)
    return CheckResult(passed=passed, checks=checks, summary=summary)
