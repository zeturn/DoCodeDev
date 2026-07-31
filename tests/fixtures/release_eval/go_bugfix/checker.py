import base64
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

# Hidden test is generated into the workspace at check time (post agent run) and
# exercises inputs not covered by the public tests. It lives in the fixture dir
# only as bytes so the Agent never sees it.
_HIDDEN_GO = (
    "package calc\n\n"
    'import "testing"\n\n'
    "func TestHiddenX(t *testing.T) {\n"
    "\tif Add(5, 5) != 10 { t.Fatal(\"Add(5,5)\") }\n"
    "\tif Add(-1, 1) != 0 { t.Fatal(\"Add(-1,1)\") }\n"
    "\tif Mul(7, 6) != 42 { t.Fatal(\"Mul(7,6)\") }\n"
    "}\n"
)
_B64 = base64.b64encode(_HIDDEN_GO.encode()).decode()

_WRITE = (
    "import base64,pathlib; "
    f"pathlib.Path('zz_hidden_x_test.go').write_text(base64.b64decode('{_B64}').decode())"
)

HIDDEN = (
    f"python3 -c \"{_WRITE}\" || python -c \"{_WRITE}\" "
    f"&& go test -run TestHiddenX -count=1 . "
    f"&& echo HIDDEN_OK"
)


async def run_check(ctx: CheckerContext) -> CheckResult:
    checks = list(await assert_required_commands(ctx))
    checks.append(await assert_tests_unmodified(ctx, ["calc_test.go"]))
    checks.append(await assert_implementation_modified(ctx, "calc.go"))
    code, out = await ctx.inspector.run_command(HIDDEN)
    checks.append(check("hidden_behavior", code == 0, "HIDDEN_OK" if code == 0 else out[:400]))
    checks.append(await assert_terminal(ctx))
    checks.append(await assert_artifact_present(ctx))
    passed = all(c.passed for c in checks)
    summary = "passed" if passed else "; ".join(c.details for c in checks if not c.passed)
    return CheckResult(passed=passed, checks=checks, summary=summary)
