"""
Category K: the relational layer for multi-table analysis (see the multi-table research plan).

Single-table tools assume one clean table. With several related tables, most errors happen
before any statistic runs, at the join, and they are silent: the join succeeds and the numbers
look plausible. These tools make each step of getting from several tables to one analysis
table explicit and checked:

    discover_tables -> infer_keys -> build_relationship_graph -> plan_join_path
        -> aggregate_to_grain (when a step would fan out) -> validate_join / execute_join_plan
        -> check_unit_of_analysis -> (statistical skills) -> detect_aggregation_reversal

Key inference is deterministic and explainable, in the spirit of HoPF (Jiang & Naumann 2020):
primary keys from unique, non-null columns; foreign keys from inclusion dependencies (child
values contained in a parent key) scored together with name evidence. Every candidate is
reported with the evidence for it, so a wrong guess can be seen and overridden.
"""
from __future__ import annotations

import dataclasses
import itertools
import pathlib
import re
import sqlite3

import numpy as np
import pandas as pd

from .common import ToolResult, is_numeric

_TABLE_SUFFIXES = (".csv", ".tsv", ".txt", ".parquet", ".pq", ".xlsx", ".xls")
_SQLITE_SUFFIXES = (".db", ".sqlite", ".sqlite3")


def _table_name(path: pathlib.Path) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", path.stem).strip("_").lower()


def _singular(table: str) -> str:
    """'orders' -> 'order', 'categories' -> 'category', 'order_items' -> 'order_item'."""
    if table.endswith("ies"):
        return table[:-3] + "y"
    if table.endswith(("ses", "xes")):
        return table[:-2]
    if table.endswith("s") and not table.endswith("ss"):
        return table[:-1]
    return table


# --------------------------------------------------------------------------- #
# discover_tables
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class TablesReport(ToolResult):
    tables: dict = dataclasses.field(default_factory=dict)          # name -> DataFrame
    schema: dict = dataclasses.field(default_factory=dict)          # name -> [columns]
    row_counts: dict = dataclasses.field(default_factory=dict)
    declared_primary_keys: dict = dataclasses.field(default_factory=dict)   # name -> [cols]
    declared_foreign_keys: list = dataclasses.field(default_factory=list)   # [{child_table, ...}]
    sources: dict = dataclasses.field(default_factory=dict)         # name -> file it came from


def discover_tables(source) -> TablesReport:
    """
    Load every table the analysis may need and say what is there before anything is joined.

    `source` can be a folder (every CSV/TSV/Parquet/Excel file in it), a list of file paths, a
    SQLite database file (all tables, plus its declared primary and foreign keys), or a dict of
    {name: DataFrame}. Table names come from file names ("Order Items.csv" -> "order_items").
    CSVs go through load_dataset, so each delimiter is sniffed rather than assumed.
    """
    from .loading import load_dataset

    tables, sources, warnings = {}, {}, []
    declared_pk, declared_fk = {}, []

    if isinstance(source, dict):
        for name, df in source.items():
            if not isinstance(df, pd.DataFrame):
                return TablesReport(ok=False, warnings=[f"{name!r} is not a DataFrame."])
            tables[str(name)] = df
            sources[str(name)] = "in-memory"
    else:
        paths = [pathlib.Path(p) for p in source] if isinstance(source, (list, tuple)) else [pathlib.Path(source)]
        files = []
        for p in paths:
            if p.is_dir():
                files += sorted(f for f in p.iterdir() if f.suffix.lower() in _TABLE_SUFFIXES + _SQLITE_SUFFIXES)
            elif p.exists():
                files.append(p)
            else:
                warnings.append(f"Not found: {p}")
        if not files:
            return TablesReport(ok=False, warnings=warnings + ["No table files found."])
        for f in files:
            if f.suffix.lower() in _SQLITE_SUFFIXES:
                con = sqlite3.connect(f)
                try:
                    names = [r[0] for r in con.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
                    for t in names:
                        tables[t] = pd.read_sql_query(f'SELECT * FROM "{t}"', con)
                        sources[t] = str(f)
                        info = con.execute(f'PRAGMA table_info("{t}")').fetchall()
                        pks = [r[1] for r in sorted(info, key=lambda r: r[5]) if r[5] > 0]
                        if pks:
                            declared_pk[t] = pks
                        for fk in con.execute(f'PRAGMA foreign_key_list("{t}")').fetchall():
                            declared_fk.append({"child_table": t, "child_column": fk[3],
                                                "parent_table": fk[2], "parent_column": fk[4]})
                finally:
                    con.close()
                continue
            name = _table_name(f)
            if name in tables:
                warnings.append(f"Two files map to table name {name!r}; kept the first.")
                continue
            rep = load_dataset(f)
            if not rep.ok:
                warnings.append(f"Skipped {f.name}: {rep.warnings}")
                continue
            warnings += [f"{f.name}: {w}" for w in rep.warnings]
            tables[name], sources[name] = rep.df, str(f)

    if not tables:
        return TablesReport(ok=False, warnings=warnings + ["No tables could be loaded."])
    schema = {t: list(df.columns) for t, df in tables.items()}
    row_counts = {t: int(len(df)) for t, df in tables.items()}
    empty = [t for t, n in row_counts.items() if n == 0]
    if empty:
        warnings.append(f"Empty table(s): {empty}.")
    if len(tables) == 1:
        warnings.append("Only one table loaded; the single-table skills apply directly.")
    return TablesReport(ok=True, warnings=warnings, tables=tables, schema=schema, row_counts=row_counts,
                        declared_primary_keys=declared_pk, declared_foreign_keys=declared_fk, sources=sources)


# --------------------------------------------------------------------------- #
# infer_keys
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class KeysReport(ToolResult):
    primary_keys: dict = dataclasses.field(default_factory=dict)       # table -> [cols] (1 col, or composite)
    pk_candidates: dict = dataclasses.field(default_factory=dict)      # table -> [{column, score, evidence}]
    foreign_keys: list = dataclasses.field(default_factory=list)       # accepted links
    rejected_candidates: list = dataclasses.field(default_factory=list)  # plausible links not accepted, with reason


def _idish(col: str) -> bool:
    c = col.lower()
    return c == "id" or c.endswith(("_id", "id", "_key", "_code", "_no", "_num"))


def _pk_score(table: str, col: str, s: pd.Series, position: int) -> tuple[float, list[str]]:
    c, ev, score = col.lower(), [], 0.0
    if c in ("id", f"{_singular(table)}_id", f"{table}_id"):
        score += 3; ev.append("name matches the table")
    elif _idish(col):
        score += 2; ev.append("ID-like name")
    if position == 0:
        score += 1; ev.append("first column")
    if pd.api.types.is_float_dtype(s):
        score -= 2; ev.append("float values (unusual for a key)")
    if pd.api.types.is_bool_dtype(s) or s.nunique() <= 2:
        score -= 3; ev.append("too few distinct values")
    return score, ev


def _key_set(s: pd.Series) -> set:
    vals = s.dropna()
    if is_numeric(vals) and len(vals) and np.all(np.mod(vals.to_numpy(float), 1) == 0):
        return set(vals.astype("int64").tolist())   # 3.0 and 3 are the same key
    return set(vals.astype(str).tolist())


def _compatible(a: pd.Series, b: pd.Series) -> bool:
    return is_numeric(a) == is_numeric(b)


def _name_score(child_col: str, parent_table: str, parent_col: str) -> tuple[int, str]:
    c, p, sing = child_col.lower(), parent_col.lower(), _singular(parent_table.lower())
    if c == p and p != "id":
        return 2, "same name as the parent key"
    if c in (f"{sing}_{p}", f"{sing}_id", f"{parent_table.lower()}_id", f"{sing}{p}"):
        return 2, "name is <parent table>_<key>"
    if sing in c and _idish(child_col):
        return 1, "name mentions the parent table"
    if _idish(child_col):
        return 0, "ID-like name, but not naming the parent"
    return 0, "no name evidence"


def infer_keys(tables: dict, declared_primary_keys: dict | None = None,
               declared_foreign_keys: list | None = None, min_inclusion: float = 0.9) -> KeysReport:
    """
    Infer primary keys and foreign keys from the data, and say why.

    Primary key: a column with no missing values and no duplicates, preferring ID-like names
    (`customer_id` in `customers`). If no single column qualifies, an ID-like pair is tried
    (e.g. order_id + line_no for order lines).

    Foreign key: a column whose values are (almost) all contained in another table's primary key
    (an inclusion dependency), with compatible types. Containment is measured over rows: the share
    of non-missing child rows whose value exists in the parent, so it equals 1 - the orphan rate.
    It must also have name evidence that points at the parent (`customer_id` -> customers), unless
    it holds 10+ distinct text IDs that are all contained (`reviewer_id` holding customer IDs).
    Small integer columns (quantities, ratings, line numbers) fit inside almost every integer ID
    range by chance, so numbers alone never make a key.
    Each child column gets at most one parent: the best-scoring one. `min_inclusion` below 1
    tolerates orphans (a few child rows whose parent is missing); validate_join counts them.

    Declared keys (from a SQLite schema) are taken as given and marked source="declared".
    """
    if not tables:
        return KeysReport(ok=False, warnings=["No tables given."])
    declared_primary_keys = declared_primary_keys or {}
    pks, pk_cands, warnings = {}, {}, []

    for t, df in tables.items():
        if t in declared_primary_keys:
            pks[t] = list(declared_primary_keys[t])
            pk_cands[t] = [{"column": "+".join(pks[t]), "score": None, "evidence": ["declared in the schema"]}]
            continue
        cands = []
        for i, col in enumerate(df.columns):
            s = df[col]
            if len(df) and s.notna().all() and s.is_unique:
                score, ev = _pk_score(t, col, s, i)
                cands.append({"column": col, "score": score, "evidence": ["unique and never missing"] + ev})
        cands.sort(key=lambda c: -c["score"])
        pk_cands[t] = cands
        if cands and cands[0]["score"] > 0:
            pks[t] = [cands[0]["column"]]
        else:
            idcols = [c for c in df.columns if _idish(c) or c.lower() in ("line", "line_no", "seq", "position")]
            pair = next((list(p) for p in itertools.combinations(idcols, 2)
                         if df[list(p)].notna().all().all() and not df.duplicated(list(p)).any()), None)
            if pair:
                pks[t] = pair
                pk_cands[t] = [{"column": "+".join(pair), "score": None, "evidence": ["unique ID-like pair"]}] + cands
            else:
                warnings.append(f"No primary key found for {t!r}; it can only be joined as a child table.")

    fks, rejected = [], []
    declared = [dict(fk, source="declared", inclusion=None, score=None, evidence=["declared in the schema"])
                for fk in (declared_foreign_keys or [])]
    declared_cols = {(fk["child_table"], fk["child_column"]) for fk in declared}

    single_pk = {t: k[0] for t, k in pks.items() if len(k) == 1}
    parent_sets = {t: _key_set(tables[t][k]) for t, k in single_pk.items()}
    for child, df in tables.items():
        for col in df.columns:
            if (child, col) in declared_cols:
                continue
            vals = _key_set(df[col])
            if not vals:
                continue
            col_vals = df[col].dropna()
            col_norm = (col_vals.astype("int64") if is_numeric(col_vals) and np.all(np.mod(col_vals.to_numpy(float), 1) == 0)
                        else col_vals.astype(str))
            options = []
            for parent, pcol in single_pk.items():
                if parent == child or not _compatible(df[col], tables[parent][pcol]):
                    continue
                if pks.get(child) == [col] and col != pcol:
                    continue  # a table's own key only links out to an identically named key (1:1 extension)
                inclusion = float(col_norm.isin(parent_sets[parent]).mean())    # share of rows with a parent
                distinct_inclusion = len(vals & parent_sets[parent]) / len(vals)
                if inclusion < 0.5:
                    continue
                ns, why = _name_score(col, parent, pcol)
                text_ids = not is_numeric(df[col]) and len(vals) >= 10 and inclusion >= 0.98
                accepted = inclusion >= min_inclusion and (ns >= 1 or text_ids)
                reason = ("" if accepted else
                          f"inclusion {inclusion:.0%} below {min_inclusion:.0%}" if inclusion < min_inclusion else
                          "values fit, but no name evidence (small integers fit most ID ranges by chance)")
                options.append({"child_table": child, "child_column": col, "parent_table": parent,
                                "parent_column": pcol, "inclusion": round(inclusion, 4),
                                "score": round(0.5 * inclusion + 0.25 * ns, 4), "accepted": accepted,
                                "evidence": [f"{inclusion:.1%} of rows ({distinct_inclusion:.0%} of {len(vals)} distinct "
                                             f"values) found in {parent}.{pcol}", why], "reason": reason})
            if not options:
                continue
            options.sort(key=lambda o: -o["score"])
            best = next((o for o in options if o["accepted"]), None)
            for o in options:
                if o is best:
                    fks.append({k: v for k, v in o.items() if k not in ("accepted", "reason")} | {"source": "inferred"})
                else:
                    why_not = o["reason"] or (f"a better-matching parent was chosen ({best['parent_table']})" if best else "")
                    rejected.append({k: v for k, v in o.items() if k not in ("accepted", "reason")} | {"reason": why_not})
    fks = declared + fks
    for fk in fks:
        if fk.get("inclusion") is not None and fk["inclusion"] < 1:
            warnings.append(f"{fk['child_table']}.{fk['child_column']}: {1 - fk['inclusion']:.1%} of its rows have no "
                            f"match in {fk['parent_table']} (orphans). A join will drop or null-fill them.")
    if not fks:
        warnings.append("No foreign keys found: the tables may be unrelated, or the keys are named or typed "
                        "differently. Pass declared keys if you know them.")
    return KeysReport(ok=bool(fks), warnings=warnings, primary_keys=pks, pk_candidates=pk_cands,
                      foreign_keys=fks, rejected_candidates=rejected)


# --------------------------------------------------------------------------- #
# build_relationship_graph
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class RelationshipGraph(ToolResult):
    tables: list = dataclasses.field(default_factory=list)
    edges: list = dataclasses.field(default_factory=list)      # {child_table, child_column, parent_table, parent_column, cardinality}
    bridges: dict = dataclasses.field(default_factory=dict)    # bridge table -> [the tables it links N:M]
    components: list = dataclasses.field(default_factory=list)  # groups of tables connected to each other
    description: list = dataclasses.field(default_factory=list)  # one plain sentence per edge


def build_relationship_graph(tables: dict, keys: KeysReport) -> RelationshipGraph:
    """
    Turn the key links into a graph: tables are nodes, foreign keys are edges labelled with their
    cardinality from the parent's side, "1:1" or "1:N" (one customer, many orders). A table with
    two or more foreign keys and no single-column key of its own is a bridge: through it, its
    parents are N:M (orders and products, via order_items). Tables in separate components cannot
    be joined at all.
    """
    if not keys.ok and not keys.foreign_keys:
        return RelationshipGraph(ok=False, warnings=["No foreign keys to build a graph from."], tables=list(tables))
    edges, sentences = [], []
    for fk in keys.foreign_keys:
        child = tables[fk["child_table"]][fk["child_column"]].dropna()
        card = "1:1" if child.is_unique else "1:N"
        edges.append({**{k: fk[k] for k in ("child_table", "child_column", "parent_table", "parent_column")},
                      "cardinality": card})
        sentences.append(f"Each {_singular(fk['parent_table'])} has "
                         f"{'at most one' if card == '1:1' else 'zero or more'} {fk['child_table']} rows "
                         f"({fk['child_table']}.{fk['child_column']} -> {fk['parent_table']}.{fk['parent_column']}).")
    bridges = {}
    for t in tables:
        parents = sorted({e["parent_table"] for e in edges if e["child_table"] == t})
        if len(parents) >= 2 and len(keys.primary_keys.get(t, [])) != 1:
            bridges[t] = parents
            sentences.append(f"{t} links {' and '.join(parents)} many-to-many.")
    parent_of = {t: t for t in tables}

    def find(x):
        while parent_of[x] != x:
            parent_of[x] = parent_of[parent_of[x]]
            x = parent_of[x]
        return x
    for e in edges:
        parent_of[find(e["child_table"])] = find(e["parent_table"])
    groups: dict = {}
    for t in tables:
        groups.setdefault(find(t), []).append(t)
    components = sorted((sorted(g) for g in groups.values()), key=len, reverse=True)
    warnings = []
    isolated = [g[0] for g in components if len(g) == 1]
    if isolated:
        warnings.append(f"Not linked to any other table: {isolated}.")
    if len(components) - len(isolated) > 1:
        warnings.append(f"The tables form {len(components)} separate groups; questions spanning groups cannot be answered by joining.")
    return RelationshipGraph(ok=True, warnings=warnings, tables=list(tables), edges=edges, bridges=bridges,
                             components=components, description=sentences)


# --------------------------------------------------------------------------- #
# plan_join_path
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class JoinPlan(ToolResult):
    base_table: str = ""
    grain: str = ""                                               # what one row of the result is
    steps: list = dataclasses.field(default_factory=list)         # {left, left_on, right, right_on, direction}
    fan_out_steps: list = dataclasses.field(default_factory=list)  # steps that multiply rows
    alternatives: dict = dataclasses.field(default_factory=dict)  # "a -> b" -> [other paths]


def _adjacency(graph: RelationshipGraph):
    adj: dict = {t: [] for t in graph.tables}
    for e in graph.edges:
        c, p = e["child_table"], e["parent_table"]
        adj[c].append((p, e["child_column"], e["parent_column"], "many_to_one"))
        adj[p].append((c, e["parent_column"], e["child_column"], "one_to_many"))
    return adj


def _paths(adj, start, goal, max_len):
    stack = [(start, [start])]
    while stack:
        node, path = stack.pop()
        if node == goal:
            yield path
            continue
        if len(path) > max_len:
            continue
        for nxt, *_ in adj[node]:
            if nxt not in path:
                stack.append((nxt, path + [nxt]))


def plan_join_path(graph: RelationshipGraph, needed_tables: list[str], base_table: str | None = None,
                   max_path_length: int = 5) -> JoinPlan:
    """
    Plan how to join the tables a question needs, starting from a base table whose rows are the
    unit of analysis (the grain). Each step is labelled many_to_one (safe: looks up one parent row
    per row) or one_to_many (fans out: each row is repeated once per child row, which inflates sums
    and means of the repeated columns). Without `base_table`, the table that reaches the others with
    the fewest fan-out steps is chosen; say which grain the question is really about and pass it.

    When two routes of similar length connect the same tables (customers reach products through
    reviews, or through orders and order_items), the shortest is planned and the others are reported
    in `alternatives`, because the routes answer different questions ("products a customer reviewed"
    vs "products a customer bought").
    """
    unknown = [t for t in needed_tables if t not in graph.tables]
    if unknown:
        return JoinPlan(ok=False, warnings=[f"Unknown table(s): {unknown}. Known: {graph.tables}"])
    if not needed_tables:
        return JoinPlan(ok=False, warnings=["Name at least one table."])
    adj = _adjacency(graph)
    edge_info = {(a, b): (lc, rc, d) for a in adj for (b, lc, rc, d) in adj[a]}

    def plan_from(base):
        steps, joined, alts = [], {base}, {}
        for target in needed_tables:
            if target in joined:
                continue
            paths = sorted(_paths(adj, base, target, max_path_length), key=len)
            if not paths:
                return None
            best = paths[0]
            # Only routes about as short as the best are real ambiguity; a 5-hop detour is not a rival.
            others = [p for p in paths[1:] if p != best and len(p) <= len(best) + 1]
            if others:
                alts[f"{base} -> {target}"] = [" -> ".join(p) for p in others[:3]]
            for a, b in zip(best, best[1:]):
                if b in joined:
                    continue
                lc, rc, d = edge_info[(a, b)]
                steps.append({"left": a, "left_on": lc, "right": b, "right_on": rc, "direction": d})
                joined.add(b)
        return steps, alts

    candidates = [base_table] if base_table else list(needed_tables)
    if base_table and base_table not in graph.tables:
        return JoinPlan(ok=False, warnings=[f"Unknown base table {base_table!r}."])
    best = None
    for b in candidates:
        r = plan_from(b)
        if r is None:
            continue
        fan = sum(s["direction"] == "one_to_many" for s in r[0])
        if best is None or fan < best[1] or (fan == best[1] and len(r[0]) < len(best[2][0])):
            best = (b, fan, r)
    if best is None:
        return JoinPlan(ok=False, warnings=[f"No join path connects {needed_tables} within {max_path_length} steps."])
    base, _, (steps, alts) = best
    fan_steps = [s for s in steps if s["direction"] == "one_to_many"]
    warnings = []
    for s in fan_steps:
        warnings.append(f"Joining {s['left']} -> {s['right']} repeats each {s['left']} row once per {s['right']} row. "
                        f"Either analyse at the {s['right']} grain, or first aggregate {s['right']} to one row per "
                        f"{s['right_on']} with aggregate_to_grain.")
    for k, v in alts.items():
        warnings.append(f"More than one route for {k}: planned the shortest; alternatives {v} would answer a "
                        "different question. Confirm which relationship the question means.")
    return JoinPlan(ok=True, warnings=warnings, base_table=base,
                    grain=f"one row per {_singular(base)} ({base})" if not fan_steps else
                    f"mixed: starts at {base}, then fans out (see warnings)",
                    steps=steps, fan_out_steps=fan_steps, alternatives=alts)


# --------------------------------------------------------------------------- #
# aggregate_to_grain
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class GrainResult(ToolResult):
    df: pd.DataFrame | None = None
    key: list = dataclasses.field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0
    new_columns: list = dataclasses.field(default_factory=list)


def aggregate_to_grain(df: pd.DataFrame, key: str | list[str], aggregations: dict,
                        count_column: str | None = "n_rows") -> GrainResult:
    """
    Roll a child table up to one row per key before joining it to its parent (order lines to
    orders: total quantity, number of lines). This is the fix for fan-out: joining the aggregate
    is many_to_one, so the parent's rows are not repeated.

    `aggregations` maps column -> function name or list of names ("sum", "mean", "max", "nunique",
    ...). New columns are named <column>_<function>.
    """
    key = [key] if isinstance(key, str) else list(key)
    missing = [c for c in key + list(aggregations) if c not in df.columns]
    if missing:
        return GrainResult(ok=False, warnings=[f"Column(s) not found: {missing}"])
    spec, names = {}, []
    for col, funcs in aggregations.items():
        for f in ([funcs] if isinstance(funcs, str) else funcs):
            name = f"{col}_{f}"
            if f in ("sum", "mean", "median", "std", "min", "max") and not is_numeric(df[col]):
                return GrainResult(ok=False, warnings=[f"{f} needs a numeric column; {col!r} is not."])
            spec[name] = (col, f)
            names.append(name)
    warnings = []
    null_keys = int(df[key].isna().any(axis=1).sum())
    if null_keys:
        warnings.append(f"{null_keys} row(s) with a missing key were left out.")
    g = df.dropna(subset=key).groupby(key, observed=True)
    out = g.agg(**spec) if spec else g.size().to_frame("_tmp").drop(columns="_tmp")
    if count_column:
        out[count_column] = g.size()
        names.append(count_column)
    out = out.reset_index()
    return GrainResult(ok=True, warnings=warnings, df=out, key=key, rows_before=len(df),
                       rows_after=len(out), new_columns=names)


# --------------------------------------------------------------------------- #
# validate_join / execute_join_plan
# --------------------------------------------------------------------------- #
_EXPECTED = {"one_to_one", "many_to_one", "one_to_many", "many_to_many"}


@dataclasses.dataclass
class JoinReport(ToolResult):
    df: pd.DataFrame | None = None
    how: str = "left"
    observed: str = ""             # relationship found in the data
    expected: str = ""
    rows_left: int = 0
    rows_right: int = 0
    rows_result: int = 0
    fan_out_factor: float = 1.0    # result rows / left rows
    left_unmatched: int = 0        # left rows with no partner (lost by inner, null-filled by left)
    left_unmatched_pct: float = 0.0
    right_unmatched: int = 0
    duplicate_left_keys: int = 0
    duplicate_right_keys: int = 0


def validate_join(left: pd.DataFrame, right: pd.DataFrame, left_on: str, right_on: str | None = None,
                  how: str = "left", expected: str = "many_to_one", left_name: str = "left",
                  right_name: str = "right", max_unmatched_pct: float = 5.0) -> JoinReport:
    """
    Do the join and report what it did to the rows, so a bad join is caught before any statistic.

    - fan_out_factor: result rows / left rows. Above 1 means left rows were repeated, so any sum or
      mean of a left-table column is now weighted by how many partners each row had.
    - left_unmatched: left rows with no partner. An inner join drops them silently (biasing the
      sample); a left join keeps them with missing right-hand values.
    - observed vs expected relationship: e.g. expected many_to_one (each order has one customer)
      but the right key is duplicated, so the join is really many_to_many.

    ok=False when the join repeats rows that were expected to stay unique, or drops/orphans more
    than `max_unmatched_pct` percent of rows.
    """
    right_on = right_on or left_on
    if expected not in _EXPECTED:
        return JoinReport(ok=False, warnings=[f"expected must be one of {sorted(_EXPECTED)}."])
    if how not in ("left", "inner", "right", "outer"):
        return JoinReport(ok=False, warnings=["how must be left, inner, right or outer."])
    for name, df, col in ((left_name, left, left_on), (right_name, right, right_on)):
        if col not in df.columns:
            return JoinReport(ok=False, warnings=[f"{name} has no column {col!r}."])
    lk, rk = left[left_on], right[right_on]
    if is_numeric(lk) != is_numeric(rk):
        return JoinReport(ok=False, warnings=[f"Key types differ ({lk.dtype} vs {rk.dtype}); "
                                              "values like 7 and '7' would never match. Convert one side first."])
    dup_l = int(lk.dropna().duplicated().sum())
    dup_r = int(rk.dropna().duplicated().sum())
    observed = {(False, False): "one_to_one", (True, False): "many_to_one",
                (False, True): "one_to_many", (True, True): "many_to_many"}[(dup_l > 0, dup_r > 0)]
    suffix = "_" + re.sub(r"\W+", "_", right_name)
    merged = left.merge(right, how=how, left_on=left_on, right_on=right_on, indicator=True,
                        suffixes=("", suffix))
    left_unmatched = int((~lk.isin(set(rk.dropna()))).sum())
    right_unmatched = int((~rk.isin(set(lk.dropna()))).sum())
    pct = round(100 * left_unmatched / len(left), 2) if len(left) else 0.0
    fan = round(len(merged) / len(left), 4) if len(left) else 1.0
    merged = merged.drop(columns="_merge")

    warnings, ok = [], True
    if observed == "many_to_many":
        ok = False
        warnings.append(f"Both keys repeat: this is a many-to-many join ({len(left)} x {len(right)} rows "
                        f"became {len(merged)}). Aggregate one side to its key first.")
    elif expected in ("many_to_one", "one_to_one") and dup_r:
        ok = False
        warnings.append(f"Expected each {left_name} row to match at most one {right_name} row, but "
                        f"{right_on!r} repeats {dup_r} time(s) in {right_name}: rows are multiplied "
                        f"(fan-out x{fan}).")
    elif fan > 1 and expected == "one_to_many":
        warnings.append(f"Fan-out x{fan}, as expected for one-to-many: the result has one row per "
                        f"{right_name} row. Do not average {left_name}-level columns over it; aggregate first or "
                        "analyse at this grain.")
    if pct > max_unmatched_pct:
        ok = False
        verb = "dropped" if how == "inner" else "kept with missing values"
        warnings.append(f"{left_unmatched} {left_name} row(s) ({pct}%) have no match in {right_name} and were {verb}.")
    elif left_unmatched:
        warnings.append(f"{left_unmatched} {left_name} row(s) ({pct}%) have no match in {right_name}.")
    if expected != observed and observed != "many_to_many":
        warnings.append(f"Expected {expected}, but the keys behave as {observed}.")
    return JoinReport(ok=ok, warnings=warnings, df=merged, how=how, observed=observed, expected=expected,
                      rows_left=len(left), rows_right=len(right), rows_result=len(merged), fan_out_factor=fan,
                      left_unmatched=left_unmatched, left_unmatched_pct=pct, right_unmatched=right_unmatched,
                      duplicate_left_keys=dup_l, duplicate_right_keys=dup_r)


@dataclasses.dataclass
class JoinRun(ToolResult):
    df: pd.DataFrame | None = None
    steps: list = dataclasses.field(default_factory=list)   # one summary per join step
    grain: str = ""


def execute_join_plan(tables: dict, plan: JoinPlan, how: str = "left",
                      columns: dict | None = None) -> JoinRun:
    """
    Run a JoinPlan step by step through validate_join, stopping at the first step that fails, and
    return the final table with a per-step record of rows before/after and unmatched rows.
    `columns` ({table: [cols]}) keeps only the columns the question needs from each table (keys are
    added automatically), which avoids name clashes and very wide tables.
    """
    if not plan.ok:
        return JoinRun(ok=False, warnings=["Refusing to run a failed plan."] + plan.warnings)
    columns = columns or {}

    def pick(t):
        df = tables[t]
        if t not in columns:
            return df
        keys = {s["left_on"] for s in plan.steps if s["left"] == t} | {s["right_on"] for s in plan.steps if s["right"] == t}
        return df[[c for c in df.columns if c in set(columns[t]) | keys]]

    current = pick(plan.base_table)
    record, warnings = [], list(plan.warnings)
    for s in plan.steps:
        expected = "many_to_one" if s["direction"] == "many_to_one" else "one_to_many"
        rep = validate_join(current, pick(s["right"]), s["left_on"], s["right_on"], how=how, expected=expected,
                            left_name=s["left"], right_name=s["right"])
        record.append({"join": f"{s['left']}.{s['left_on']} -> {s['right']}.{s['right_on']}", "observed": rep.observed,
                       "rows_before": rep.rows_left, "rows_after": rep.rows_result,
                       "fan_out_factor": rep.fan_out_factor, "left_unmatched": rep.left_unmatched, "ok": rep.ok})
        warnings += [f"[{s['left']} -> {s['right']}] {w}" for w in rep.warnings]
        if not rep.ok:
            return JoinRun(ok=False, warnings=warnings, df=rep.df, steps=record, grain=plan.grain)
        current = rep.df
    return JoinRun(ok=True, warnings=warnings, df=current, steps=record, grain=plan.grain)


# --------------------------------------------------------------------------- #
# check_unit_of_analysis
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class UnitReport(ToolResult):
    entity: str = ""
    n_rows: int = 0
    n_entities: int = 0
    max_rows_per_entity: int = 0
    mean_rows_per_entity: float = 0.0
    repeated: bool = False
    entity_level_columns: list = dataclasses.field(default_factory=list)  # constant within each entity


def check_unit_of_analysis(df: pd.DataFrame, entity_col: str, columns: list[str] | None = None) -> UnitReport:
    """
    Check whether one entity (a customer, an order) now spans several rows, as it does after a
    one-to-many join. If it does, rows are not independent: a t-test or correlation at row level
    counts an entity with 10 rows ten times. `entity_level_columns` lists the columns that are
    constant within each entity (attributes of the entity, e.g. customer age): analyse those after
    aggregating to one row per entity.
    """
    if entity_col not in df.columns:
        return UnitReport(ok=False, warnings=[f"Column {entity_col!r} not found."], entity=entity_col)
    sizes = df.groupby(entity_col, observed=True).size()
    cols = [c for c in (columns or df.columns) if c != entity_col and c in df.columns]
    repeated = bool((sizes > 1).any())
    level = []
    if repeated:
        multi = df[df[entity_col].isin(sizes[sizes > 1].index)]
        for c in cols:
            if multi.groupby(entity_col, observed=True)[c].nunique(dropna=False).max() <= 1:
                level.append(c)
    warnings = []
    if repeated:
        warnings.append(f"{len(df)} rows describe {len(sizes)} {entity_col} values (up to {int(sizes.max())} rows each). "
                        "Row-level tests treat repeated rows as independent observations, which they are not. "
                        f"Aggregate to one row per {entity_col} (aggregate_to_grain), or use a method that accounts "
                        "for grouping.")
        if level:
            warnings.append(f"{level} are constant within each {entity_col}: they are {entity_col}-level attributes, "
                            f"so analyse them at the {entity_col} level.")
    return UnitReport(ok=not repeated, warnings=warnings, entity=entity_col, n_rows=len(df), n_entities=len(sizes),
                      max_rows_per_entity=int(sizes.max()) if len(sizes) else 0,
                      mean_rows_per_entity=round(float(sizes.mean()), 3) if len(sizes) else 0.0,
                      repeated=repeated, entity_level_columns=level)


# --------------------------------------------------------------------------- #
# detect_aggregation_reversal
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class ReversalReport(ToolResult):
    measure: str = ""                    # "difference in means (b - a)" or "Pearson r"
    pooled_effect: float = 0.0
    within_effect: float = 0.0           # size-weighted average of the per-group effects
    group_effects: dict = dataclasses.field(default_factory=dict)   # group -> {effect, n}
    reversed: bool = False
    n_groups_opposite: int = 0


def detect_aggregation_reversal(df: pd.DataFrame, x: str, y: str, group_col: str,
                                min_group_size: int = 10) -> ReversalReport:
    """
    Compare the association of x with y overall (pooled) against within each level of group_col.
    A reversal (Simpson's paradox) is when the pooled association points one way and the
    within-group associations the other: the grouping variable is confounding the pooled result
    (Pearl 2014). Joins make this easy to miss, because the confounder often sits in another table.

    x with two levels: difference in mean y between the levels (works for 0/1 outcomes too: a
    difference in rates). x numeric: Pearson r between x and y.
    """
    missing = [c for c in (x, y, group_col) if c not in df.columns]
    if missing:
        return ReversalReport(ok=False, warnings=[f"Column(s) not found: {missing}"])
    sub = df[[x, y, group_col]].dropna()
    if not is_numeric(sub[y]):
        return ReversalReport(ok=False, warnings=[f"{y!r} must be numeric (use 0/1 for a yes/no outcome)."])
    levels = sorted(sub[x].unique(), key=str)
    if len(levels) == 2:
        a, b = levels
        measure = f"difference in mean {y} ({x}={b} minus {x}={a})"

        def effect(d):
            ga, gb = d.loc[d[x] == a, y], d.loc[d[x] == b, y]
            return None if len(ga) < 2 or len(gb) < 2 else float(gb.mean() - ga.mean())
    elif is_numeric(sub[x]):
        measure = f"Pearson r between {x} and {y}"

        def effect(d):
            return None if len(d) < 3 or d[x].std() == 0 or d[y].std() == 0 else float(d[x].corr(d[y]))
    else:
        return ReversalReport(ok=False, warnings=[f"{x!r} must be numeric or have exactly two levels."])

    pooled = effect(sub)
    if pooled is None:
        return ReversalReport(ok=False, warnings=["Not enough data for the pooled comparison."])
    groups, weights, vals = {}, [], []
    for g, d in sub.groupby(group_col, observed=True):
        e = effect(d) if len(d) >= min_group_size else None
        groups[str(g)] = {"effect": e, "n": int(len(d))}
        if e is not None:
            vals.append(e)
            weights.append(len(d))
    if not vals:
        return ReversalReport(ok=False, warnings=[f"No group has {min_group_size}+ rows with both {x} levels."])
    within = float(np.average(vals, weights=weights))
    tiny = 1e-9
    opposite = sum(1 for v in vals if v * pooled < 0)
    reversed_ = abs(pooled) > tiny and abs(within) > tiny and pooled * within < 0
    warnings = []
    if reversed_:
        warnings.append(f"Aggregation reversal (Simpson's paradox): overall the {measure} is {pooled:+.3g}, but within "
                        f"{group_col} groups it averages {within:+.3g} ({opposite} of {len(vals)} groups point the other "
                        f"way). {group_col} confounds the overall comparison; report the within-group result.")
    elif opposite:
        warnings.append(f"{opposite} of {len(vals)} {group_col} groups point the opposite way to the overall result; "
                        "check whether the effect depends on the group.")
    return ReversalReport(ok=True, warnings=warnings, measure=measure, pooled_effect=pooled, within_effect=within,
                          group_effects=groups, reversed=reversed_, n_groups_opposite=opposite)
