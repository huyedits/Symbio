"""Goals: written only by /goal, served as the user's own words, and handed
to the "am I finished?" check."""
from symbio import constants
from symbio.app import goals


def test_goals_are_added_listed_ticked_and_cleared():
    assert goals.list_goals() == []
    assert "Goal 1 set" in goals.add_goal("ship the server kit")
    assert "Goal 2 set" in goals.add_goal("measure the continue check")
    assert "Already a goal" in goals.add_goal("Ship the server kit")
    assert [g.split(" (set ")[0] for g in goals.list_goals()] == [
        "ship the server kit", "measure the continue check"]
    assert "Done: ship the server kit" in goals.done_goal(1)
    assert "No goal 5" in goals.done_goal(5)
    assert "Cleared 1" in goals.clear_goals()
    assert not constants.GOALS_FILE.exists()


def test_goals_are_bounded():
    for i in range(goals.LIMIT):
        goals.add_goal(f"goal number {i}")
    assert "already open" in goals.add_goal("one too many")
    assert "under" in goals.add_goal("x" * (goals.CHAR_LIMIT + 1))
    goals.clear_goals()


def test_the_prompt_lists_open_goals_last_and_only_when_there_are_some():
    assert goals.goals_block("Huy") == ""
    goals.add_goal("finish the routing eval")
    block = goals.goals_block("Huy")
    assert "Open goals Huy set" in block and "1. finish the routing eval" in block
    assert "(set " not in block
    goals.clear_goals()


def test_writing_the_goals_file_with_a_tool_is_flagged_sensitive():
    from symbio import safety
    for name in ("goals.md", "standing_instructions.md"):
        risk = safety._assess_file_risk("write_file", {"path": name, "content": "x"})
        assert "sensitive_file" in risk["flags"] and risk["risk_score"] == 3
