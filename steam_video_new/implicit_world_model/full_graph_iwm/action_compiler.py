"""Deterministic single-cursor legal action compilation and real execution."""

from __future__ import annotations

from dataclasses import replace
import hashlib

from memory_graph.types import MemoryNode

from .contracts import (
    ActionKind,
    AnswerabilityState,
    CursorBeliefState,
    GraphActionExecution,
    LegalGraphAction,
    RetainedEvidenceGraph,
)


class GraphActionCompiler:
    """Compile every executable action without question-conditioned pruning."""

    def compile(
        self,
        belief: CursorBeliefState,
        graph: RetainedEvidenceGraph,
    ) -> tuple[LegalGraphAction, ...]:
        visible = {node.node_id: node for node in visible_graph_nodes(graph)}
        if belief.current_node_id is not None and belief.current_node_id not in visible:
            raise ValueError("current cursor is not visible in the retained graph")
        if belief.remaining_reads == 0:
            return self._terminal_actions(belief)

        actions: list[LegalGraphAction] = []
        acquired = set(belief.acquired_evidence)
        current = belief.current_node_id
        if current is None:
            for node_id in sorted(visible):
                actions.append(
                    _action(
                        ActionKind.START_AT,
                        target_id=node_id,
                        reads_evidence=True,
                    )
                )
            actions.extend(self._terminal_actions(belief))
            return _deduplicate(actions)

        for edge in graph.temporal_edges:
            if edge.src == current and edge.dst in visible and edge.dst not in acquired:
                actions.append(
                    _action(
                        ActionKind.TEMPORAL_FORWARD,
                        source_id=current,
                        target_id=edge.dst,
                        edge_id=edge.edge_id,
                        relation=edge.relation,
                        reads_evidence=True,
                    )
                )
            elif (
                edge.dst == current and edge.src in visible and edge.src not in acquired
            ):
                actions.append(
                    _action(
                        ActionKind.TEMPORAL_BACKWARD,
                        source_id=current,
                        target_id=edge.src,
                        edge_id=edge.edge_id,
                        relation=edge.relation,
                        reads_evidence=True,
                    )
                )

        for edge in graph.correlation_edges:
            if current not in {edge.src, edge.dst}:
                continue
            if not edge.evidence_refs:
                continue
            if edge.src == current:
                target = edge.dst
                affinity = edge.src_to_dst_affinity
            else:
                target = edge.src
                affinity = edge.dst_to_src_affinity
            if affinity <= 0.0:
                continue
            if target not in visible or target in acquired:
                continue
            actions.append(
                _action(
                    ActionKind.FOLLOW_CORRELATION,
                    source_id=current,
                    target_id=target,
                    edge_id=edge.edge_id,
                    relation=edge.channel,
                    reads_evidence=True,
                )
            )

        for edge in graph.candidate_edges:
            if current not in {edge.src, edge.dst}:
                continue
            target = edge.dst if edge.src == current else edge.src
            if not edge.permits(current, target):
                continue
            if target not in visible or target in acquired:
                continue
            actions.append(
                _action(
                    ActionKind.FOLLOW_CORRELATION,
                    source_id=current,
                    target_id=target,
                    edge_id=edge.edge_id,
                    relation=edge.channel,
                    reads_evidence=True,
                )
            )

        for node_id in belief.acquired_evidence:
            if node_id != current and node_id in visible:
                actions.append(
                    _action(
                        ActionKind.BACKTRACK,
                        source_id=current,
                        target_id=node_id,
                        reads_evidence=False,
                    )
                )
        actions.extend(self._terminal_actions(belief))
        return _deduplicate(actions)

    @staticmethod
    def _terminal_actions(belief: CursorBeliefState) -> tuple[LegalGraphAction, ...]:
        actions = [
            _action(ActionKind.STOP),
            _action(ActionKind.ABSTAIN),
        ]
        if belief.answerability is AnswerabilityState.READY:
            actions.append(_action(ActionKind.ANSWER))
        return tuple(actions)


def execute_graph_action(
    action: LegalGraphAction,
    belief: CursorBeliefState,
    graph: RetainedEvidenceGraph,
) -> GraphActionExecution:
    """Execute one real graph operation; imagined observations never enter here."""

    legal = {
        value.action_id: value for value in GraphActionCompiler().compile(belief, graph)
    }
    if action.action_id not in legal or legal[action.action_id] != action:
        raise ValueError("action is not legal in the current graph state")
    if action.kind in {ActionKind.STOP, ActionKind.ANSWER, ActionKind.ABSTAIN}:
        return GraphActionExecution(action, None, belief, reread=False)

    assert action.target_id is not None
    node_by_id = graph.node_by_id
    if action.target_id not in node_by_id:
        raise ValueError("action target is absent from retained memory")
    previous_cursor = belief.current_node_id
    if action.kind is ActionKind.BACKTRACK:
        updated = replace(
            belief,
            belief_id=f"{belief.belief_id}:step:{belief.step + 1}",
            current_node_id=action.target_id,
            cursor_history=_append_cursor(belief.cursor_history, previous_cursor),
            step=belief.step + 1,
        )
        return GraphActionExecution(action, None, updated, reread=False)

    already_acquired = action.target_id in set(belief.acquired_evidence)
    acquired = tuple(dict.fromkeys((*belief.acquired_evidence, action.target_id)))
    imagined = tuple(
        node_id for node_id in belief.imagined_evidence if node_id != action.target_id
    )
    updated = replace(
        belief,
        belief_id=f"{belief.belief_id}:step:{belief.step + 1}",
        current_node_id=action.target_id,
        acquired_evidence=acquired,
        imagined_evidence=imagined,
        cursor_history=_append_cursor(belief.cursor_history, previous_cursor),
        remaining_reads=max(0, belief.remaining_reads - 1),
        step=belief.step + 1,
    )
    return GraphActionExecution(
        action=action,
        observation=node_by_id[action.target_id],
        updated_belief=updated,
        reread=already_acquired,
    )


def visible_graph_nodes(graph: RetainedEvidenceGraph) -> tuple[MemoryNode, ...]:
    """Return the exact node set shared by action and model-input compilation."""

    return tuple(node for node in graph.nodes if _node_is_visible(node, graph))


def _node_is_visible(node: object, graph: RetainedEvidenceGraph) -> bool:
    metadata = getattr(node, "metadata", {})
    if metadata.get("hidden_supervision") is True:
        return False
    visibility = metadata.get("visibility")
    if isinstance(visibility, dict) and (
        visibility.get("hidden_supervision") is True
        or visibility.get("visible_to_agent") is False
    ):
        return False
    if str(visibility or "").lower() in {"hidden", "future"}:
        return False
    horizon = graph.metadata.get("observation_end_s")
    if horizon is not None and getattr(node, "time_span").end_s > float(horizon) + 1e-6:
        return False
    return True


def _action(
    kind: ActionKind,
    *,
    source_id: str | None = None,
    target_id: str | None = None,
    edge_id: str | None = None,
    relation: str | None = None,
    reads_evidence: bool = False,
) -> LegalGraphAction:
    payload = "\x1f".join(
        (kind.value, source_id or "", target_id or "", edge_id or "", relation or "")
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return LegalGraphAction(
        action_id=f"action:{digest}",
        kind=kind,
        source_id=source_id,
        target_id=target_id,
        edge_id=edge_id,
        relation=relation,
        reads_evidence=reads_evidence,
    )


def _deduplicate(actions: list[LegalGraphAction]) -> tuple[LegalGraphAction, ...]:
    selected = {action.action_id: action for action in actions}
    return tuple(
        sorted(
            selected.values(),
            key=lambda action: (
                action.kind in {ActionKind.STOP, ActionKind.ANSWER, ActionKind.ABSTAIN},
                action.kind.value,
                action.target_id or "",
                action.edge_id or "",
            ),
        )
    )


def _append_cursor(history: tuple[str, ...], cursor: str | None) -> tuple[str, ...]:
    return history if cursor is None else tuple((*history, cursor))
