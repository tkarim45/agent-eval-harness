"""Key-free tests: scorer correctness + the harness discriminates good from bad agents."""
from ageval.harness import run
from ageval.scorer import answer_correct, step_efficiency, tool_prf


def test_answer_correct_numeric_tolerant():
    assert answer_correct("the total is 75.5", "75.5", "numeric")
    assert answer_correct("75.50", "75.5", "numeric")
    assert not answer_correct("76", "75.5", "numeric")


def test_answer_correct_text_substring():
    assert answer_correct("the manager is carol", "carol", "text")
    assert not answer_correct("dan", "carol", "text")


def test_tool_prf_multiset():
    prf = tool_prf(["catalog_lookup", "catalog_lookup", "calculator"],
                   ["catalog_lookup", "catalog_lookup", "calculator"])
    assert prf["f1"] == 1.0
    prf2 = tool_prf(["catalog_lookup"], ["catalog_lookup", "catalog_lookup", "calculator"])
    assert prf2["recall"] < 1.0 and prf2["precision"] == 1.0


def test_step_efficiency_caps_at_one():
    assert step_efficiency(1, 3) == 1.0      # fewer steps than optimal still caps at 1
    assert step_efficiency(6, 3) == 0.5      # took twice as many
    assert step_efficiency(0, 3) == 0.0


def test_heuristic_solves_benchmark():
    report = run("heuristic")
    assert report["summary"]["success_rate"] == 1.0
    assert report["summary"]["mean_tool_f1"] == 1.0


def test_harness_discriminates_naive_below_heuristic():
    good = run("heuristic")["summary"]
    bad = run("naive")["summary"]
    assert bad["success_rate"] < good["success_rate"]
    assert bad["mean_tool_f1"] < good["mean_tool_f1"]
    assert bad["mean_step_efficiency"] <= good["mean_step_efficiency"]
