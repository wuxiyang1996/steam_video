"""Deterministic legal actions from a sparse local proposal neighborhood."""

from __future__ import annotations

import hashlib

from .contracts import (
    NavigationAction,
    NavigationActionKind,
    NavigationGraph,
)


def compile_legal_actions(
    graph: NavigationGraph,
    *,
    cursor_id: str | None,
    acquired_ids: tuple[str, ...],
    entry_node_ids: tuple[str, ...] | None = None,
) -> tuple[NavigationAction, ...]:
    acquired = set(acquired_ids)
    actions: list[NavigationAction] = []
    if cursor_id is None:
        entries = graph.entry_node_ids if entry_node_ids is None else entry_node_ids
        if not entries:
            return (_terminal(NavigationActionKind.ABSTAIN),)
        for target in entries:
            if target not in set(graph.node_ids):
                raise ValueError("entry action targets an unknown address")
            if target not in acquired:
                actions.append(
                    _read(NavigationActionKind.START, None, target, None, None)
                )
    else:
        for proposal in graph.neighbors(cursor_id):
            target = proposal.dst if proposal.src == cursor_id else proposal.src
            if target in acquired or not proposal.permits(cursor_id, target):
                continue
            actions.append(
                _read(
                    NavigationActionKind.FOLLOW,
                    cursor_id,
                    target,
                    proposal.proposal_id,
                    proposal.kind,
                )
            )
    actions.extend(
        (_terminal(NavigationActionKind.STOP), _terminal(NavigationActionKind.ABSTAIN))
    )
    return _deduplicate(actions)


def _read(kind, source, target, proposal_id, proposal_kind) -> NavigationAction:
    identity = "\x1f".join(
        (kind.value, source or "", target, proposal_id or "", str(proposal_kind or ""))
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
    return NavigationAction(
        action_id=f"navigation-action:{digest}",
        kind=kind,
        source_id=source,
        target_id=target,
        proposal_id=proposal_id,
        proposal_kind=proposal_kind,
        reads_evidence=True,
    )


def _terminal(kind: NavigationActionKind) -> NavigationAction:
    return NavigationAction(f"navigation-action:{kind.value}", kind)


def _deduplicate(actions: list[NavigationAction]) -> tuple[NavigationAction, ...]:
    return tuple({action.action_id: action for action in actions}.values())
