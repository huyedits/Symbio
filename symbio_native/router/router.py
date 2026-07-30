"""Inference-time router that selects relevant substrate branches."""

from __future__ import annotations

from ..substrate.branch import Branch


class Router:
    """Select branches by token overlap between the current input and branch
    trigger tokens, or by explicit branch name hints in the prompt.
    """

    def __init__(self, branches: list[Branch] | None = None):
        self.branches = branches or []

    def add(self, branch: Branch) -> None:
        self.branches.append(branch)

    def select(self, prompt_ids: list[int], *, top_k: int = 3) -> list[Branch]:
        """Return the most relevant branches for this prompt.

        Scoring is intersection size of prompt token ids and branch trigger ids,
        divided by the number of trigger ids (recall against the trigger set).
        """
        prompt_set = set(prompt_ids)
        scored: list[tuple[float, Branch]] = []
        for branch in self.branches:
            if not branch.trigger_ids:
                continue
            overlap = len(prompt_set & set(branch.trigger_ids))
            score = overlap / len(branch.trigger_ids)
            if score > 0:
                scored.append((score, branch))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in scored[:top_k]]
