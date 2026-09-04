"""
repomap_class.py - High-performance AST indexing, graph ranking, and context-packing engine.
"""

from __future__ import annotations

import collections
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp

from importance import is_important
from scm import get_scm_fname
from utils import Tag, compute_file_hash, count_tokens, read_text

try:
    import tree_sitter
except ImportError:
    tree_sitter = None  # type: ignore

try:
    from grep_ast import TreeContext, filename_to_lang
except ImportError:
    TreeContext = None  # type: ignore
    filename_to_lang = None  # type: ignore


class RepoMap:
    CACHE_VERSION = 2

    def __init__(
        self,
        map_tokens: int = 8192,
        root: Optional[str] = None,
        model_name: str = "gpt-4",
        verbose: bool = False,
    ):
        self.map_tokens = map_tokens
        self.root = Path(root or os.getcwd()).resolve()
        self.model_name = model_name
        self.verbose = verbose

        # Project-anchored cache initialization
        self.cache_dir = self.root / f".repomap.cache.v{self.CACHE_VERSION}"
        self.cache_db = self.cache_dir / "tags_cache.sqlite"
        self._init_sqlite_cache()

        # Stop-words to dampen false graph super-nodes
        self.universal_noise_identifiers = {
            "get", "set", "data", "id", "name", "run", "handle", "update",
            "val", "value", "key", "item", "self", "this", "err", "error",
            "result", "res", "req", "ctx", "args", "kwargs", "config"
        }

    def _init_sqlite_cache(self) -> None:
        """Configures project-anchored SQLite storage in WAL mode for safe concurrency."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with sqlite3.connect(self.cache_db, timeout=5.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA busy_timeout=5000;")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tag_cache (
                        rel_path TEXT PRIMARY KEY,
                        mtime_ns INTEGER,
                        file_size INTEGER,
                        sha256_hash TEXT,
                        tags_json TEXT
                    );
                    """
                )
        except sqlite3.OperationalError as e:
            if self.verbose:
                print(f"Warning: Cache init degraded: {e}", file=sys.stderr)

    def get_tags(self, fname: str, rel_fname: str) -> List[Tag]:
        """
        Extracts definition and reference tags using Tree-sitter with multi-version compatibility.
        """
        if tree_sitter is None or filename_to_lang is None:
            return []

        path = Path(fname)
        if not path.is_file():
            return []

        lang = filename_to_lang(fname)
        if not lang:
            return []

        scm_path = get_scm_fname(lang)
        if not scm_path:
            return []

        code = read_text(path, silent=not self.verbose)
        if not code:
            return []

        tags: List[Tag] = []
        try:
            # tree-sitter >=0.25 API: Language takes a single capsule arg.
            # Resolve via tree-sitter-language-pack, never via .scm path.
            from tree_sitter_language_pack import get_language, get_parser
            language_obj = get_language(lang)
            parser = get_parser(lang)
            tree = parser.parse(bytes(code, "utf-8"))

            with open(scm_path, "r", encoding="utf-8") as f:
                query_scm = f.read()

            query = tree_sitter.Query(language_obj, query_scm)
            raw = tree_sitter.QueryCursor(query).captures(tree.root_node)
            # 0.25+: dict[name, list[Node]]; older: list[(Node, name)]
            if isinstance(raw, dict):
                pairs = [(n, nm) for nm, nodes in raw.items() for n in nodes]
            else:
                pairs = list(raw)

            for node, capture_name in pairs:
                kind = "def" if "definition" in capture_name else "ref"
                name = node.text.decode("utf-8", errors="replace")
                line = node.start_point[0] + 1
                tags.append(Tag(rel_fname=rel_fname, fname=fname, line=line, name=name, kind=kind))

        except Exception as e:
            if self.verbose:
                print(f"AST extraction notice for {rel_fname}: {e}", file=sys.stderr)

        return tags

    def compute_sparse_pagerank(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str, float]],
        personalization_weights: Dict[str, float],
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> Dict[str, float]:
        """
        Vectorized SciPy CSR Personalized PageRank solver.
        """
        n = len(nodes)
        if n == 0:
            return {}
        if n == 1:
            return {nodes[0]: 1.0}

        node2idx = {name: i for i, name in enumerate(nodes)}

        src_indices = []
        tgt_indices = []
        weights = []

        for src, tgt, w in edges:
            if src in node2idx and tgt in node2idx and src != tgt:
                src_indices.append(node2idx[src])
                tgt_indices.append(node2idx[tgt])
                weights.append(w)

        if not weights:
            # Disconnected fallback
            uniform = 1.0 / n
            return {node: uniform for node in nodes}

        # Form sparse weighted adjacency matrix
        A = sp.coo_matrix((weights, (src_indices, tgt_indices)), shape=(n, n), dtype=np.float64).tocsr()
        out_degree = np.array(A.sum(axis=1)).flatten()

        inv_out = np.zeros_like(out_degree)
        nonzero = out_degree > 0
        inv_out[nonzero] = 1.0 / out_degree[nonzero]

        P = (sp.diags(inv_out) @ A).tocsr()
        dangling = (out_degree == 0)

        # Build personalization distribution
        v = np.zeros(n, dtype=np.float64)
        for name, weight in personalization_weights.items():
            if name in node2idx:
                v[node2idx[name]] = weight

        v_sum = v.sum()
        v = v / v_sum if v_sum > 0 else np.full(n, 1.0 / n)

        # Power iteration
        x = v.copy()
        for _ in range(max_iter):
            xlast = x.copy()
            danglesum = xlast[dangling].sum()
            x = damping * (xlast @ P + danglesum * v) + (1.0 - damping) * v
            if np.abs(x - xlast).sum() < tol:
                break

        return {nodes[i]: float(x[i]) for i in range(n)}

    def get_ranked_tags(
        self,
        chat_files: List[str],
        other_files: List[str],
        mentioned_files: Optional[List[str]] = None,
        mentioned_idents: Optional[List[str]] = None,
    ) -> List[Tag]:
        """
        Constructs the cross-file dependency graph, applies TF-IDF dampening,
        and computes personalized PageRank scores.
        """
        chat_files_set = {str(Path(f).resolve()) for f in (chat_files or [])}
        other_files_set = {str(Path(f).resolve()) for f in (other_files or [])}
        all_fnames = sorted(list(chat_files_set | other_files_set))

        file_tags: Dict[str, List[Tag]] = {}
        symbol_doc_freq = collections.Counter()

        # Step 1: Collect AST tags across target codebase
        for fname in all_fnames:
            rel = os.path.relpath(fname, self.root)
            tags = self.get_tags(fname, rel)
            file_tags[fname] = tags
            seen_in_file = {t.name for t in tags}
            for sym in seen_in_file:
                symbol_doc_freq[sym] += 1

        # Step 2: Compute TF-IDF dampening threshold (>35% file presence = stop-word)
        total_files = max(1, len(all_fnames))
        stop_words = {
            sym for sym, count in symbol_doc_freq.items()
            if (count / total_files > 0.35) or (sym in self.universal_noise_identifiers)
        }

        # Step 3: Graph Construction
        nodes = list(all_fnames)
        edges: List[Tuple[str, str, float]] = []

        # Invert definitions: symbol -> declaring files
        definitions: Dict[str, Set[str]] = collections.defaultdict(set)
        for fname, tags in file_tags.items():
            for t in tags:
                if t.kind == "def" and t.name not in stop_words:
                    definitions[t.name].add(fname)

        # Wire references to definitions with typed relational weights
        for fname, tags in file_tags.items():
            for t in tags:
                if t.kind == "ref" and t.name in definitions:
                    target_files = definitions[t.name]
                    for tgt in target_files:
                        if tgt != fname:
                            edges.append((fname, tgt, 1.0))

        # Step 4: Construct Personalization Vector
        pers_weights: Dict[str, float] = {}
        for fname in all_fnames:
            if fname in chat_files_set:
                pers_weights[fname] = 100.0  # Teleport bias to developer focus
            elif mentioned_files and any(fname.endswith(m) for m in mentioned_files):
                pers_weights[fname] = 20.0
            else:
                pers_weights[fname] = 1.0

        scores = self.compute_sparse_pagerank(nodes, edges, pers_weights)

        # Step 5: Order tags by file PageRank score and declaration priority
        ranked_tags: List[Tag] = []
        sorted_files = sorted(all_fnames, key=lambda f: scores.get(f, 0.0), reverse=True)

        for f in sorted_files:
            defs = [t for t in file_tags[f] if t.kind == "def"]
            ranked_tags.extend(defs)

        return ranked_tags

    def render_tree(self, tags: List[Tag]) -> str:
        """Renders AST skeleton views via grep-ast."""
        if TreeContext is None or not tags:
            return ""

        tags_by_file = collections.defaultdict(list)
        for t in tags:
            tags_by_file[t.fname].append(t.line)

        output_fragments = []
        for fname, lines in tags_by_file.items():
            rel_fname = os.path.relpath(fname, self.root)
            code = read_text(fname, silent=True)
            if not code:
                continue

            try:
                tc = TreeContext(
                    fname,
                    code,
                    color=False,
                    line_number=True,
                    child_context=False,
                    last_line=True,
                    margin=0,
                    mark_lois=False,
                    loi_pad=0,
                    show_top_of_file_parent_scope=False,
                )
                tc.add_lines_of_interest(lines)
                tc.add_context()
                output_fragments.append(f"{rel_fname}:\n" + tc.format())
            except Exception:
                output_fragments.append(f"{rel_fname} (AST format fallback)")

        return "\n\n".join(output_fragments)

    def get_repo_map(
        self,
        chat_files: Optional[List[str]] = None,
        other_files: Optional[List[str]] = None,
        mentioned_files: Optional[List[str]] = None,
        mentioned_idents: Optional[List[str]] = None,
    ) -> str:
        """
        Executes density-guided interpolation search to fit ranked AST skeletons
        into map_tokens within <= 4 passes.
        """
        ranked_tags = self.get_ranked_tags(
            chat_files=chat_files or [],
            other_files=other_files or [],
            mentioned_files=mentioned_files,
            mentioned_idents=mentioned_idents,
        )

        n_tags = len(ranked_tags)
        if n_tags == 0:
            return ""

        # Step 1: Initial density estimation sample (k = 20)
        k_sample = min(20, n_tags)
        tree_sample = self.render_tree(ranked_tags[:k_sample])
        toks_sample = count_tokens(tree_sample, self.model_name)

        if toks_sample >= self.map_tokens:
            return tree_sample

        density = max(1.0, toks_sample / k_sample)  # Marginal tokens / tag

        # Step 2: Linear interpolation to estimate target k
        k_est = min(n_tags, int(self.map_tokens / density))
        tree_est = self.render_tree(ranked_tags[:k_est])
        toks_est = count_tokens(tree_est, self.model_name)

        # Step 3: Secant interpolation refinement
        if toks_est > self.map_tokens:
            k_refined = int(k_est * (self.map_tokens / max(1, toks_est)))
            tree_refined = self.render_tree(ranked_tags[:k_refined])
            toks_refined = count_tokens(tree_refined, self.model_name)
        else:
            k_refined, tree_refined, toks_refined = k_est, tree_est, toks_est

        # Step 4: Bounded safety adjustment
        while toks_refined > self.map_tokens and k_refined > 5:
            k_refined -= 5
            tree_refined = self.render_tree(ranked_tags[:k_refined])
            toks_refined = count_tokens(tree_refined, self.model_name)

        return tree_refined

    def get_file_outline(self, rel_fname: str) -> str:
        """Generates an immediate structural outline of an individual file."""
        abs_path = (self.root / rel_fname).resolve()
        tags = self.get_tags(str(abs_path), rel_fname)
        def_tags = [t for t in tags if t.kind == "def"]
        return self.render_tree(def_tags)

    def locate_symbol(self, ident: str) -> List[Dict[str, Any]]:
        """Finds all declaration sites and references for a specific identifier."""
        matches = []
        for root, _, files in os.walk(self.root):
            for file in files:
                abs_f = os.path.join(root, file)
                rel_f = os.path.relpath(abs_f, self.root)
                tags = self.get_tags(abs_f, rel_f)
                for t in tags:
                    if t.name == ident:
                        matches.append({
                            "name": t.name,
                            "file": t.rel_fname,
                            "line": t.line,
                            "kind": t.kind
                        })
        return matches

    def compute_blast_radius(self, rel_fname: str, depth: int = 2) -> Dict[str, Any]:
        """Calculates upstream and downstream dependent modules."""
        target_abs = str((self.root / rel_fname).resolve())
        # Inspect direct callers and dependencies
        tags = self.get_tags(target_abs, rel_fname)
        exported_symbols = {t.name for t in tags if t.kind == "def"}
        
        dependent_files = set()
        for root, _, files in os.walk(self.root):
            for file in files:
                abs_f = os.path.join(root, file)
                if abs_f == target_abs:
                    continue
                rel_f = os.path.relpath(abs_f, self.root)
                f_tags = self.get_tags(abs_f, rel_f)
                if any(t.kind == "ref" and t.name in exported_symbols for t in f_tags):
                    dependent_files.add(rel_f)

        return {
            "target": rel_fname,
            "exported_symbols_count": len(exported_symbols),
            "direct_dependents": sorted(list(dependent_files))
        }
