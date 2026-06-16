# Testing
**Analysis Date:** 2026-06-16

## Test Framework

The project uses Python's built-in **`unittest`** framework. No `pytest`, `nose`, or other third-party test framework is configured. There is no coverage configuration (no `.coveragerc`, no `[tool.coverage]` in `pyproject.toml`).

All test files live under the `tests/` directory and each file tests a specific source module:

```
tests/
  test_alfred_scene.py          — tests src/alfred_scene.py
  test_context_builder.py       — tests src/context_builder.py
  test_eb_agent_prompts.py      — tests src/eb_agent.py (prompt content)
  test_env_injector.py          — tests src/env_injector.py
  test_error_handling.py        — tests src/branch_runner.py (error flows)
  test_task_conditions_alfred.py — tests src/task_conditions.py
  test_training_prompt.py       — tests scripts/finetune_qwen3vl.py
  test_vlm_client.py            — tests src/vlm_client.py
```

### How to Run Tests

Run all tests from the project root:

```bash
python -m unittest discover tests/
```

Run a single test file:

```bash
python tests/test_vlm_client.py
```

Run a specific test class or method:

```bash
python -m unittest tests.test_context_builder.ContextBuilderTest
python -m unittest tests.test_env_injector.EnvInjectorTest.test_occlude_uses_single_object_placement_not_set_object_poses
```

Each test file can also be run directly since they include the standard boilerplate:

```python
if __name__ == "__main__":
    unittest.main()
```

## Test Structure

### Test Class Pattern

Every test class inherits from `unittest.TestCase` and is named after the module it tests, suffixed with `Test`:

```python
# tests/test_context_builder.py
import unittest

class ContextBuilderTest(unittest.TestCase):
    ...

# tests/test_error_handling.py
import unittest

class ErrorHandlingTest(unittest.TestCase):
    ...

# tests/test_vlm_client.py
import unittest

class VLMClientTest(unittest.TestCase):
    ...

# tests/test_env_injector.py
import unittest

class EnvInjectorTest(unittest.TestCase):
    ...

# tests/test_eb_agent_prompts.py
import unittest

class EBAgentPromptTest(unittest.TestCase):
    ...
```

Each test file contains exactly one test class and follows this convention: one class per module-under-test, with related tests grouped under it.

### Test Method Naming

Test methods use descriptive snake_case names starting with `test_`:

```python
def test_vlm_json_parse_failure_raises_instead_of_pass_fallback(self):
    ...

def test_vlm_missing_required_fields_retries_with_field_error(self):
    ...

def test_eb_history_uses_all_steps_without_oracle_fields(self):
    ...

def test_oracle_history_keeps_fork_field_names(self):
    ...

def test_branch_history_includes_parent_shared_steps_before_fork_steps(self):
    ...

def test_current_failed_step_is_rendered_for_diagnosis(self):
    ...
```

Names describe the behavior being tested rather than just the method name.

## Setup / Teardown

Tests do not use `setUp()` or `tearDown()` methods. Test fixtures are created inline within each test method using helper functions, factories, or `tempfile.TemporaryDirectory()`. Some tests use `try/finally` to monkey-patch and restore module-level globals:

```python
# tests/test_error_handling.py
with tempfile.TemporaryDirectory() as tmp:
    branch_runner_module.EnvController = FakeInitialTrapEnv
    alfred_parser.load_traj = fake_load_traj
    alfred_parser.extract_metadata = fake_extract_metadata
    try:
        with self.assertRaises(RuntimeError):
            branch_runner_module.run_single_branch(...)
        with open(f"{tmp}/task/failures_main.jsonl", encoding="utf-8") as f:
            log = f.read()
        self.assertIn("initial_trap_setup_failed", log)
    finally:
        branch_runner_module.EnvController = original_env
        alfred_parser.load_traj = original_load
        alfred_parser.extract_metadata = original_extract
```

## Mock / Fake Pattern

The project does not use `unittest.mock` or any mocking library. Instead, it uses **hand-written fakes** that implement the same duck-typed interface as the real objects. This pattern is used for controllers, agents, oracles, and environments:

### Fake Environment/Controller

```python
# tests/test_error_handling.py
class FailingMoveEnv:
    def __init__(self):
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def get_state_snapshot(self):
        return {
            "frame": self.frame,
            "metadata": {"objects": [], "agent": {"position": {}, "rotation": {"y": 0}}},
            "task_state": {},
        }

    def step(self, action, **params):
        if action in ("RotateLeft", "RotateRight", "Pass"):
            return {"success": True, "error": None, "frame": self.frame,
                    "metadata": {"objects": [], "agent": {"position": {}, "rotation": {"y": 0}}}}
        return {"success": False, "error": "blocked by obstacle", "frame": self.frame,
                "metadata": {"objects": [], "agent": {"position": {}, "rotation": {"y": 0}}}}

# tests/test_alfred_scene.py
class FakeController:
    def __init__(self, metadata=None):
        self.calls = []
        self.metadata = metadata or {"lastActionSuccess": True, "objects": []}

    def step(self, action=None, **params):
        call = dict(params)
        if isinstance(action, dict):
            call.update(action)
        elif action is not None:
            call["action"] = action
        self.calls.append(call)
        return FakeEvent(self.metadata)

class FakeEvent:
    def __init__(self, metadata):
        self.metadata = metadata
        self.frame = None
```

### Fake Agent

```python
# tests/test_error_handling.py
class InvalidActionAgent:
    def plan_intent(self, **kwargs):
        return {"intent": "approach Table", "target": "Table", "reasoning": "test"}

    def review_actions(self, **kwargs):
        return {"approved": True, "reason": "ok", "corrected_actions": []}

    def propose_action(self, **kwargs):
        self.last_errors.append(kwargs.get("last_error"))
        if len(self.last_errors) == 1:
            return {"action": "Fly", "params": {}, "reasoning": "invalid action"}
        return {"action": "MoveAhead", "params": {}, "reasoning": "valid retry"}
```

### Fake Oracle

```python
# tests/test_error_handling.py
class NoopOracle:
    def decide_injection(self, **kwargs):
        return {"inject": False, "reasoning": "", "injection": None}

    def evaluate_failure(self, **kwargs):
        return {
            "diagnosis_correct": True,
            "ground_truth": "The path is blocked.",
            "counterfactual_grade": "WA",
            "counterfactual_gold": None,
            "recovery_verdict": "unrecoverable",
            "should_fork": False,
            "fork_reasoning": None,
        }
```

### Capturing Client (to inspect VLM prompts)

```python
# tests/test_eb_agent_prompts.py
class CapturingClient:
    def __init__(self):
        self.user_text = None

    def chat_with_image_json(self, *, user_text, **kwargs):
        self.user_text = user_text
        return {
            "diagnosis": "I failed.",
            "recovery_reasoning": "I will recover.",
            "counterfactual": None,
            "proposed_recovery_action": {"action": "MoveAhead", "params": {}},
        }

    def chat_with_images_json(self, *, user_text, **kwargs):
        self.user_text = user_text
        return {"action": "MoveAhead", "params": {}, "reasoning": "I move."}
```

### Monkey-patching method replacement

For testing VLMClient's internal retry logic, individual methods are replaced directly on instances:

```python
# tests/test_error_handling.py
def test_vlm_json_parse_failure_raises_instead_of_pass_fallback(self):
    client = VLMClient("openai", "fake")

    def fake_chat_text(system_prompt, user_text, json_mode=False, temperature=0.2, max_tokens=2048):
        prompts.append(user_text)
        return "not json"

    client.chat_text = fake_chat_text  # monkey-patch instance method

    with self.assertRaises(ValueError):
        client.chat_text_json("sys", "user", max_retries=2)

    self.assertEqual(2, len(prompts))
    self.assertIn("previous response was not valid JSON", prompts[1])
```

## Test Helper Functions

Module-level helper functions create test fixtures. They are plain functions (not methods), placed at the top of the test file before the test class:

```python
# tests/test_context_builder.py
def make_step(idx, branch="main", success=True, **extra):
    step = {
        "step_id": f"s{idx}",
        "branch_id": branch,
        "parent_step_id": f"s{idx - 1}" if idx > 0 else None,
        "step_index_in_branch": idx,
        "action": "MoveAhead",
        "action_params": {"moveMagnitude": 0.25, "objectId": "Hidden|0|0|0"},
        "success": success,
        "error_message": None if success else "blocked",
        "eb_reasoning": f"reason {idx}",
        ...
    }
    step.update(extra)
    return step

def json_lines(context):
    return [json.loads(line) for line in context.splitlines() if line.startswith("{")]

# tests/test_task_conditions_alfred.py
def make_obj(object_id, object_type, *, pickupable=False, receptacle=False,
             toggleable=False, visible=True, ...):
    obj = {
        "objectId": object_id,
        "objectType": object_type,
        "pickupable": pickupable,
        "receptacle": receptacle,
        ...
    }
    return obj

# tests/test_error_handling.py
def _obj(object_id, object_type, **extra):
    data = {"objectId": object_id, "objectType": object_type, "visible": True, ...}
    data.update(extra)
    return data
```

## Assertion Patterns

### assertEqual (expected == actual)

Used as the primary assertion for value comparisons:

```python
self.assertEqual(12, len(rows))
self.assertEqual(0, rows[0]["step"])
self.assertEqual("MoveAhead", result["action"])
self.assertEqual("PickupObject", rows[-1]["action"])
self.assertEqual(("main", "s0"), (s["branch_id"], s["step_id"]))
self.assertEqual(2, len(prompts))
```

### assertTrue / assertFalse

Used for boolean conditions and membership checks that evaluate to bool:

```python
self.assertTrue(remove_result["success"])
self.assertTrue(close_result["success"])
self.assertFalse(ok)
self.assertFalse(rows[-1]["success"])
```

### assertIn / assertNotIn

Used abundantly for prompt content verification (checking that key text is present or absent):

```python
self.assertIn("HAND STATUS: holding AlarmClock (AlarmClock|1)", client.user_text)
self.assertIn("TASK COMPLETION CRITERIA", client.user_text)
self.assertIn("Full EB history:", prompt)
self.assertIn("reason 0", context)
self.assertNotIn("oracle", context)
self.assertNotIn("Hidden|0|0|0", context)
self.assertNotIn("previous response was not valid JSON", prompts[1])
self.assertNotIn("scene/goal/plan", prompt)
```

### assertRaises (exception testing)

Used with context manager syntax:

```python
with self.assertRaises(ValueError):
    client.chat_text_json("sys", "user", max_retries=2)

with self.assertRaises(RuntimeError):
    branch_runner_module.run_single_branch(...)
```

## Test Pattern: Prompt Content Verification

Several tests verify that prompt-building functions include the right contextual information. The pattern is:
1. Create a capturing client or override methods
2. Call the agent method
3. Assert on the captured prompt text

```python
# tests/test_eb_agent_prompts.py
def test_phase3_prompt_includes_shared_task_context(self):
    client = CapturingClient()
    agent = EBAgent(client)

    agent.diagnose_failure(
        task_goal="move the alarm clock from one desk to another one",
        error_message="InvalidOperationException",
        image=np.zeros((1, 1, 3)),
        action_history=[],
        cascade_level=1,
        visible_objects=_visible_alarm_clock(),
        inventory_objects=[{"objectId": "AlarmClock|1", "objectType": "AlarmClock"}],
        task_criteria="  - AlarmClock must be inside Desk",
    )

    self.assertIn("HAND STATUS: holding AlarmClock (AlarmClock|1)", client.user_text)
    self.assertIn("TASK COMPLETION CRITERIA", client.user_text)
    self.assertIn("AlarmClock must be inside Desk", client.user_text)
```

## Test Pattern: Action Trace Verification

Tests for ALFRED scene restoration verify the exact sequence of AI2-THOR controller calls:

```python
# tests/test_alfred_scene.py
def test_restore_uses_official_alfred_order(self):
    controller = FakeController({...})
    scene = {...}

    restore_alfred_scene(controller, scene)

    self.assertEqual(
        [
            "Initialize",
            "Pass",
            "ToggleObjectOff",
            "Pass",
            "DirtyObject",
            "EmptyLiquidFromObject",
            "Pass",
            "SetObjectPoses",
        ],
        [c["action"] for c in controller.calls],
    )
```

## Test Pattern: Agent Behavior Simulation

For testing error handling flows, fakes simulate multi-step agent behavior using internal counters:

```python
# tests/test_error_handling.py
class DoneRejectedAgent:
    def plan_intent(self, **kwargs):
        self.last_errors.append(kwargs.get("last_error"))
        if len(self.last_errors) == 1:
            return {"intent": "approach Table", ...}
        if len(self.last_errors) == 2:
            return {"intent": "Done", ...}
        return {"intent": "approach Table", ...}  # continue after rejection
```

## Coverage

No coverage configuration file exists (no `.coveragerc`, no `[tool.coverage]` section in `pyproject.toml`). To measure coverage, use:

```bash
pip install coverage
coverage run -m unittest discover tests/
coverage report -m
```
