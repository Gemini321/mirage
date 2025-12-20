#!/usr/bin/env python3
"""
Visualize task dependencies in a Mirage MPK task graph JSON.

The task graph JSON schema (as emitted by Mirage) contains:
  - all_tasks: list of tasks; each task has 'dependent_event' and 'trigger_event'
  - all_events: list of events; each event has 'event_type', 'first_task_id', 'last_task_id', 'num_triggers'
  - first_tasks: list of initial task indices

We visualize a bipartite dependency graph:
  - Edge: event -> task    (task waits on dependent_event)
  - Edge: task  -> event   (task triggers trigger_event)

By default, we render only a bounded neighborhood (BFS) around the initial tasks
to keep graphs readable. Use --full to render everything (can be huge).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import DefaultDict, Iterable, Literal


INVALID_EVENT_ID = 9223372036854775806  # EVENT_INVALID_ID (0x7ffffffffffffffe)


@dataclass(frozen=True)
class Node:
    kind: Literal["task", "event"]
    idx: int


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _parse_enum_names(header_path: str, enum_name: str) -> dict[int, str]:
    """
    Parse C++ enum entries like: TASK_LINEAR = 120,
    Returns {120: "TASK_LINEAR"}.
    """
    if not os.path.exists(header_path):
        return {}
    text = open(header_path, "r", encoding="utf-8").read().splitlines()
    in_enum = False
    out: dict[int, str] = {}
    # Very small, robust-enough parser for this header style.
    entry_re = re.compile(r"^\s*([A-Z0-9_]+)\s*=\s*([0-9]+)\s*,?\s*$")
    for line in text:
        if not in_enum:
            if re.search(rf"\benum\s+{re.escape(enum_name)}\b", line):
                in_enum = True
            continue
        if "};" in line or line.strip() == "}":
            in_enum = False
            continue
        m = entry_re.match(line)
        if not m:
            continue
        name, value = m.group(1), int(m.group(2))
        out[value] = name
    return out


def _safe_label(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _build_bipartite_index(
    tasks: list[dict],
    num_events: int,
) -> tuple[DefaultDict[int, list[int]], DefaultDict[int, list[int]]]:
    """
    Returns:
      - triggers_by_event[e] = [task_idx, ...] with task.trigger_event == e
      - dependents_by_event[e] = [task_idx, ...] with task.dependent_event == e
    """
    triggers_by_event: DefaultDict[int, list[int]] = defaultdict(list)
    dependents_by_event: DefaultDict[int, list[int]] = defaultdict(list)
    for task_idx, t in enumerate(tasks):
        te = t.get("trigger_event", INVALID_EVENT_ID)
        if te != INVALID_EVENT_ID and isinstance(te, int) and 0 <= te < num_events:
            triggers_by_event[te].append(task_idx)
        de = t.get("dependent_event", INVALID_EVENT_ID)
        if de != INVALID_EVENT_ID and isinstance(de, int) and 0 <= de < num_events:
            dependents_by_event[de].append(task_idx)
    return triggers_by_event, dependents_by_event


def _neighbors(
    node: Node,
    tasks: list[dict],
    triggers_by_event: DefaultDict[int, list[int]],
    dependents_by_event: DefaultDict[int, list[int]],
) -> Iterable[Node]:
    if node.kind == "task":
        t = tasks[node.idx]
        te = t.get("trigger_event", INVALID_EVENT_ID)
        de = t.get("dependent_event", INVALID_EVENT_ID)
        if te != INVALID_EVENT_ID:
            yield Node("event", int(te))
        if de != INVALID_EVENT_ID:
            yield Node("event", int(de))
    else:
        for ti in triggers_by_event.get(node.idx, []):
            yield Node("task", ti)
        for ti in dependents_by_event.get(node.idx, []):
            yield Node("task", ti)


def _select_subgraph(
    starts: list[Node],
    tasks: list[dict],
    triggers_by_event: DefaultDict[int, list[int]],
    dependents_by_event: DefaultDict[int, list[int]],
    max_nodes: int,
    hops: int,
) -> set[Node]:
    seen: set[Node] = set()
    q: deque[tuple[Node, int]] = deque()
    for s in starts:
        seen.add(s)
        q.append((s, 0))
    while q and len(seen) < max_nodes:
        node, depth = q.popleft()
        if depth >= hops:
            continue
        for nb in _neighbors(node, tasks, triggers_by_event, dependents_by_event):
            if nb in seen:
                continue
            seen.add(nb)
            if len(seen) >= max_nodes:
                break
            q.append((nb, depth + 1))
    return seen


def _write_dot(
    out_dot: str,
    nodes: set[Node],
    tasks: list[dict],
    events: list[dict],
    task_name_map: dict[int, str],
    event_name_map: dict[int, str],
    include_launch_edges: bool,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_dot)) or ".", exist_ok=True)
    node_ids: dict[Node, str] = {}
    for n in nodes:
        node_ids[n] = ("t" if n.kind == "task" else "e") + str(n.idx)

    lines: list[str] = []
    lines.append("digraph task_graph {")
    lines.append('  rankdir="LR";')
    lines.append('  graph [fontsize=12, fontname="sans-serif"];')
    lines.append('  node  [fontsize=10, fontname="sans-serif"];')
    lines.append('  edge  [fontsize=9, fontname="sans-serif"];')

    # Nodes
    for n in sorted(nodes, key=lambda x: (x.kind, x.idx)):
        if n.kind == "task":
            t = tasks[n.idx]
            ttype = t.get("task_type", -1)
            tname = task_name_map.get(ttype, f"TASK_{ttype}")
            label = f"task {n.idx}\\n{tname}\\nvariant={t.get('variant_id', '?')}"
            lines.append(
                f'  {node_ids[n]} [shape=box, style="rounded,filled", fillcolor="#dbe8f5", label="{_safe_label(label)}"];'
            )
        else:
            e = events[n.idx]
            etype = e.get("event_type", -1)
            ename = event_name_map.get(etype, f"EVENT_{etype}")
            label = (
                f"event {n.idx}\\n{ename}\\ntriggers={e.get('num_triggers','?')}\\n"
                f"range=[{e.get('first_task_id','?')},{e.get('last_task_id','?')})"
            )
            lines.append(
                f'  {node_ids[n]} [shape=ellipse, style="filled", fillcolor="#e0edd5", label="{_safe_label(label)}"];'
            )

    # Edges (dependencies)
    for n in nodes:
        if n.kind != "task":
            continue
        t = tasks[n.idx]
        te = t.get("trigger_event", INVALID_EVENT_ID)
        de = t.get("dependent_event", INVALID_EVENT_ID)
        if te != INVALID_EVENT_ID:
            ev = Node("event", int(te))
            if ev in nodes:
                lines.append(f"  {node_ids[n]} -> {node_ids[ev]};")
        if de != INVALID_EVENT_ID:
            ev = Node("event", int(de))
            if ev in nodes:
                lines.append(f"  {node_ids[ev]} -> {node_ids[n]};")

    # Optional: event launch edges (event -> tasks in [first,last))
    if include_launch_edges:
        for n in nodes:
            if n.kind != "event":
                continue
            e = events[n.idx]
            first = int(e.get("first_task_id", 0) or 0)
            last = int(e.get("last_task_id", 0) or 0)
            for ti in range(first, min(last, len(tasks))):
                tn = Node("task", ti)
                if tn in nodes:
                    lines.append(
                        f'  {node_ids[n]} -> {node_ids[tn]} [style=dashed, color="#888888"];'
                    )

    lines.append("}")
    with open(out_dot, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _render_dot(dot_path: str, out_path: str) -> None:
    fmt = out_path.rsplit(".", 1)[-1].lower()
    if fmt == "dot":
        return
    subprocess.check_call(["dot", f"-T{fmt}", dot_path, "-o", out_path])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Visualize dependencies in a Mirage MPK task graph JSON.",
    )
    ap.add_argument(
        "input",
        nargs="?",
        default="./task_graph_0.json",
        help="Path to task_graph_*.json (default: ./task_graph_0.json).",
    )
    ap.add_argument(
        "--out",
        default="task_graph_deps.svg",
        help="Output path (.svg/.pdf/.png/.dot). Default: task_graph_deps.svg",
    )
    ap.add_argument(
        "--max-nodes",
        type=int,
        default=400,
        help="Max nodes (tasks+events) in rendered subgraph (ignored with --full).",
    )
    ap.add_argument(
        "--hops",
        type=int,
        default=6,
        help="BFS hops from starts (ignored with --full).",
    )
    ap.add_argument(
        "--start-task",
        action="append",
        type=int,
        default=None,
        help="Start BFS from a specific task index (repeatable). Default: uses first_tasks from JSON.",
    )
    ap.add_argument(
        "--event-idx",
        type=int,
        default=None,
        help=(
            "Render the full subgraph of tasks launched by a specific event index "
            "(i.e., tasks in [event.first_task_id, event.last_task_id))."
        ),
    )
    ap.add_argument(
        "--event-expand",
        type=int,
        default=1,
        help=(
            "When using --event-idx, additionally include each selected task's "
            "dependent/trigger events (1) and optionally one more hop of tasks (2+)."
        ),
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Render the full graph (can be very large).",
    )
    ap.add_argument(
        "--include-launch-edges",
        action="store_true",
        help="Also draw dashed event->task edges for event launch ranges (can get very dense).",
    )
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        g = json.load(f)

    tasks: list[dict] = g["all_tasks"]
    events: list[dict] = g["all_events"]
    first_tasks: list[int] = g.get("first_tasks", [])

    root = _repo_root()
    header = os.path.join(root, "include/mirage/persistent_kernel/runtime_header.h")
    task_name_map = _parse_enum_names(header, "TaskType")
    event_name_map = _parse_enum_names(header, "EventType")

    triggers_by_event, dependents_by_event = _build_bipartite_index(
        tasks, len(events)
    )

    if args.full:
        nodes: set[Node] = {Node("task", i) for i in range(len(tasks))} | {
            Node("event", i) for i in range(len(events))
        }
    elif args.event_idx is not None:
        if not (0 <= args.event_idx < len(events)):
            raise SystemExit(f"--event-idx out of range: {args.event_idx} (num_events={len(events)})")
        e = events[args.event_idx]
        first = int(e.get("first_task_id", 0) or 0)
        last = int(e.get("last_task_id", 0) or 0)
        base_nodes: set[Node] = {Node("event", args.event_idx)}
        for ti in range(max(0, first), min(last, len(tasks))):
            base_nodes.add(Node("task", ti))
        if args.event_expand <= 0:
            nodes = base_nodes
        else:
            nodes = _select_subgraph(
                starts=list(base_nodes),
                tasks=tasks,
                triggers_by_event=triggers_by_event,
                dependents_by_event=dependents_by_event,
                max_nodes=max(args.max_nodes, len(base_nodes)),
                hops=args.event_expand,
            )
    else:
        if args.start_task is not None:
            starts = [Node("task", i) for i in args.start_task]
        else:
            starts = (
                [Node("task", i) for i in first_tasks] if first_tasks else [Node("task", 0)]
            )
        nodes = _select_subgraph(
            starts=starts,
            tasks=tasks,
            triggers_by_event=triggers_by_event,
            dependents_by_event=dependents_by_event,
            max_nodes=args.max_nodes,
            hops=args.hops,
        )

    out = os.path.abspath(args.out)
    dot_path = out if out.lower().endswith(".dot") else out + ".dot"
    _write_dot(
        out_dot=dot_path,
        nodes=nodes,
        tasks=tasks,
        events=events,
        task_name_map=task_name_map,
        event_name_map=event_name_map,
        include_launch_edges=args.include_launch_edges,
    )
    _render_dot(dot_path, out)

    print(f"Wrote: {dot_path}")
    if not out.lower().endswith(".dot"):
        print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
