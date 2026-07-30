"""Check the guardrail's SDK surface against the installed SDK — design §9, §12.

§9 claims the read-only property is structural: no built-in tool is registered,
only four MCP tools are allowed, a callback and a hook deny everything else, and
no looser settings are inherited. Every one of those claims is a *keyword
argument to someone else's dataclass*. If `setting_sources` were renamed
upstream, `ClaudeAgentOptions` would raise and the failure would be loud — but
if it were quietly deprecated into a no-op, the guarantee would evaporate while
`tests/test_agent.py` stayed green, because those tests pin `guardrail_spec()`,
which is our own dictionary and knows nothing about the SDK.

§12 says these symbols are to be verified against the installed SDK rather than
trusted from memory. This is that verification, executable: it binds the real
constructors with the real arguments and never opens a connection. It skips when
the optional `agent` extra is absent, so the stdlib-only suite still runs
anywhere; CI installs the extra so the check actually gates.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import inspect

import pytest

from triage import agent as agent_mod

# Skipped at run time rather than at import time, so the suite collects the same
# number of tests everywhere and the count the docs quote does not depend on
# which optional extras happen to be installed.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("claude_agent_sdk") is None,
    reason="the optional `agent` extra is not installed (pip install -e '.[agent]')",
)


@pytest.fixture(scope="module")
def sdk():
    return importlib.import_module("claude_agent_sdk")


def test_every_symbol_the_shell_binds_still_exists(sdk):
    """The names §9 and §12 list, checked as names rather than quoted as prose."""
    for name in ("create_sdk_mcp_server", "tool", "ClaudeAgentOptions", "ClaudeSDKClient",
                 "AssistantMessage", "ToolUseBlock", "ResultMessage", "HookMatcher",
                 "PermissionResultAllow", "PermissionResultDeny"):
        assert hasattr(sdk, name), f"claude-agent-sdk no longer exports {name!r}"


def test_the_options_object_still_accepts_every_field_the_policy_sets(sdk):
    """Each field here *is* one of the §9 guarantees; a dropped one is a dropped
    guarantee, not a cosmetic API change."""
    fields = {f.name for f in dataclasses.fields(sdk.ClaudeAgentOptions)}
    required = {
        "tools",              # §9: emptying the base tool set
        "allowed_tools",      # §9: the allowlist
        "disallowed_tools",   # §9: belt-and-suspenders denial
        "permission_mode",    # §9: never bypassPermissions
        "setting_sources",    # §9: inherit no looser project/user settings
        "can_use_tool",       # §9: the deny-by-default callback
        "hooks",              # §9: the PreToolUse guard
        "mcp_servers", "system_prompt", "model", "max_turns",
    }
    missing = sorted(required - fields)
    assert not missing, (
        f"ClaudeAgentOptions no longer takes {missing} — design §9 describes a "
        f"guardrail the SDK can no longer be asked for"
    )


def test_the_permission_results_take_the_arguments_the_guard_passes(sdk):
    """`can_use_tool` returns these two; a changed signature would make every
    denial a TypeError at the moment the guard is most needed."""
    inspect.signature(sdk.PermissionResultAllow).bind()
    inspect.signature(sdk.PermissionResultDeny).bind(message="denied", interrupt=False)


def test_the_mcp_server_and_tool_decorator_take_the_calls_the_shell_makes(sdk):
    inspect.signature(sdk.create_sdk_mcp_server).bind(
        name="triage", version="0.1.0", tools=[])
    inspect.signature(sdk.tool).bind("classify_incident", "description", {})


def test_the_result_message_still_carries_the_cost_accounting_seven_depends_on(sdk):
    """§7 cross-checks the SDK's own cost against ours, and §8 folds the
    orchestrator's spend into the incident budget. Both read these fields."""
    fields = {f.name for f in dataclasses.fields(sdk.ResultMessage)}
    for name in ("total_cost_usd", "usage", "model_usage", "num_turns",
                 "duration_ms", "is_error"):
        assert name in fields, f"ResultMessage no longer carries {name!r}"


def test_the_policy_we_pin_offline_is_the_policy_the_sdk_receives(sdk):
    """`tests/test_agent.py` asserts against `guardrail_spec()`. That is only
    meaningful if the spec's keys are really the options' keys."""
    fields = {f.name for f in dataclasses.fields(sdk.ClaudeAgentOptions)}
    unknown = sorted(set(agent_mod.guardrail_spec()) - fields)
    assert not unknown, (
        f"guardrail_spec() declares {unknown}, which ClaudeAgentOptions does not "
        f"accept — the offline tests would be pinning a policy that never applies"
    )
