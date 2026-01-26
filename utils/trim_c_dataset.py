#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

C_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "case",
    "do",
    "else",
    "goto",
    "typedef",
    "struct",
    "enum",
    "union",
}


def c2rust_prj_name(c_name: str) -> str:
    project_name = c_name.replace("-", "_")
    if project_name and project_name[0].isdigit():
        project_name = "proj_" + project_name
    if "." in project_name:
        project_name = project_name.split(".")[0]
    return project_name


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def strip_rust_comments(text: str) -> str:
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            out.append(" ")
            out.append(" ")
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            if i + 1 < n:
                out.append(" ")
                out.append(" ")
                i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def parse_impl_type(fragment: str) -> Optional[str]:
    # Parse type name from an impl fragment. This is conservative.
    frag = fragment.strip()
    if not frag.startswith("impl"):
        return None
    rest = frag[4:].strip()
    # Skip generics
    if rest.startswith("<"):
        depth = 0
        i = 0
        while i < len(rest):
            if rest[i] == "<":
                depth += 1
            elif rest[i] == ">":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
        rest = rest[i:].strip()
    # Collect identifiers until '{' or 'where'
    tokens = []
    i = 0
    while i < len(rest):
        if rest[i] == "{":
            break
        if rest.startswith("where", i):
            break
        if rest[i].isalnum() or rest[i] == "_":
            j = i
            while j < len(rest) and (rest[j].isalnum() or rest[j] == "_"):
                j += 1
            tokens.append(rest[i:j])
            i = j
            continue
        i += 1
    if not tokens:
        return None
    if "for" in tokens:
        # Use the identifier after the last 'for'
        for_idx = len(tokens) - 1 - tokens[::-1].index("for")
        if for_idx + 1 < len(tokens):
            return tokens[for_idx + 1]
    return tokens[0]


@dataclass
class RustSymbol:
    kind: str  # "free" or "method"
    name: str
    impl_type: Optional[str]
    file_base: str


def parse_rust_interface(path: str) -> List[RustSymbol]:
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    text = strip_rust_comments(text)
    symbols: List[RustSymbol] = []

    brace_depth = 0
    extern_c_level = 0
    impl_stack: List[Tuple[str, int]] = []
    pending_impl: Optional[str] = None

    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        # Skip string and char literals
        if ch in ('"', "'"):
            quote = ch
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        # Detect impl
        if text.startswith("impl", i) and (i == 0 or not text[i - 1].isalnum()):
            end = i
            while end < n and text[end] != "{":
                if text[end] == "\n":
                    break
                end += 1
            impl_frag = text[i:end]
            impl_type = parse_impl_type(impl_frag)
            if impl_type:
                pending_impl = impl_type
            i += 4
            continue
        # Detect pub fn
        if text.startswith("pub", i) and (i == 0 or not text[i - 1].isalnum()):
            j = i + 3
            while j < n and text[j].isspace():
                j += 1
            if text.startswith("fn", j):
                k = j + 2
                while k < n and text[k].isspace():
                    k += 1
                name_start = k
                while k < n and (text[k].isalnum() or text[k] == "_"):
                    k += 1
                fn_name = text[name_start:k]
                if fn_name:
                    if impl_stack:
                        symbols.append(
                            RustSymbol(
                                kind="method",
                                name=fn_name,
                                impl_type=impl_stack[-1][0],
                                file_base=os.path.splitext(os.path.basename(path))[0],
                            )
                        )
                    else:
                        symbols.append(
                            RustSymbol(
                                kind="free",
                                name=fn_name,
                                impl_type=None,
                                file_base=os.path.splitext(os.path.basename(path))[0],
                            )
                        )
                i = k
                continue
        # Track braces
        if ch == "{":
            brace_depth += 1
            if pending_impl:
                impl_stack.append((pending_impl, brace_depth))
                pending_impl = None
            i += 1
            continue
        if ch == "}":
            brace_depth -= 1
            while impl_stack and brace_depth < impl_stack[-1][1]:
                impl_stack.pop()
            i += 1
            continue
        i += 1

    return symbols


@dataclass
class CFunctionDef:
    name: str
    start: int
    end: int
    body: str
    path: str


def scan_c_functions(text: str, path: str) -> List[CFunctionDef]:
    funcs: List[CFunctionDef] = []
    n = len(text)
    i = 0
    brace_depth = 0
    extern_c_level = 0

    def skip_string(idx: int, quote: str) -> int:
        idx += 1
        while idx < n:
            if text[idx] == "\\":
                idx += 2
                continue
            if text[idx] == quote:
                return idx + 1
            idx += 1
        return idx

    macro_continuation = False
    while i < n:
        ch = text[i]
        # Preprocessor lines and macro continuations
        if i == 0 or text[i - 1] == "\n":
            if macro_continuation:
                line_end = text.find("\n", i)
                if line_end == -1:
                    line_end = n
                line = text[i:line_end]
                macro_continuation = line.rstrip().endswith("\\")
                i = line_end + 1
                continue
            if ch == "#":
                line_end = text.find("\n", i)
                if line_end == -1:
                    line_end = n
                line = text[i:line_end]
                macro_continuation = line.rstrip().endswith("\\")
                i = line_end + 1
                continue
        # Comments
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        # Strings
        if ch in ('"', "'"):
            i = skip_string(i, ch)
            continue

        if ch == "{":
            # detect extern "C" blocks
            if brace_depth == 0:
                line_start = text.rfind("\n", 0, i) + 1
                line = text[line_start:i]
                if re.search(r'extern\s+"C"\s*$', line.strip()):
                    extern_c_level += 1
            brace_depth += 1
            i += 1
            continue
        if ch == "}":
            brace_depth -= 1
            if extern_c_level > 0 and brace_depth < extern_c_level:
                extern_c_level -= 1
            i += 1
            continue

        at_top_level = brace_depth == 0 or (extern_c_level > 0 and brace_depth == extern_c_level)

        if at_top_level and ch == "(":
            # find identifier before '('
            j = i - 1
            while j >= 0 and text[j].isspace():
                j -= 1
            end_name = j
            while j >= 0 and (text[j].isalnum() or text[j] == "_"):
                j -= 1
            name = text[j + 1 : end_name + 1]
            if not name or name in C_KEYWORDS:
                i += 1
                continue
            # find matching ')'
            k = i + 1
            paren_depth = 1
            while k < n and paren_depth > 0:
                if text[k] == "\"":
                    k = skip_string(k, '"')
                    continue
                if text[k] == "'":
                    k = skip_string(k, "'")
                    continue
                if text[k] == "/" and k + 1 < n and text[k + 1] == "/":
                    while k < n and text[k] != "\n":
                        k += 1
                    continue
                if text[k] == "/" and k + 1 < n and text[k + 1] == "*":
                    k += 2
                    while k + 1 < n and not (text[k] == "*" and text[k + 1] == "/"):
                        k += 1
                    k += 2
                    continue
                if text[k] == "(":
                    paren_depth += 1
                elif text[k] == ")":
                    paren_depth -= 1
                k += 1
            if paren_depth != 0:
                i += 1
                continue
            # skip whitespace and attributes to find '{'
            m = k
            while m < n and text[m].isspace():
                m += 1
            if m >= n or text[m] != "{":
                i += 1
                continue

            # function definition found; find declaration start without crossing includes
            start = j + 1
            while start > 0:
                line_start = text.rfind("\n", 0, start - 1) + 1
                line = text[line_start:start]
                stripped = line.strip()
                if stripped.startswith("#"):
                    break
                if ";" in line or "}" in line:
                    break
                start = line_start
            while start < n and text[start].isspace():
                start += 1

            # find end of function body
            body_start = m
            m += 1
            depth = 1
            while m < n and depth > 0:
                if text[m] == "\"":
                    m = skip_string(m, '"')
                    continue
                if text[m] == "'":
                    m = skip_string(m, "'")
                    continue
                if text[m] == "/" and m + 1 < n and text[m + 1] == "/":
                    while m < n and text[m] != "\n":
                        m += 1
                    continue
                if text[m] == "/" and m + 1 < n and text[m + 1] == "*":
                    m += 2
                    while m + 1 < n and not (text[m] == "*" and text[m + 1] == "/"):
                        m += 1
                    m += 2
                    continue
                if text[m] == "{":
                    depth += 1
                elif text[m] == "}":
                    depth -= 1
                m += 1
            end = m
            body = text[body_start:end]
            funcs.append(CFunctionDef(name=name, start=start, end=end, body=body, path=path))
            i = end
            continue

        i += 1

    return funcs


def collect_c_functions(c_proj: str) -> Tuple[Dict[str, List[CFunctionDef]], Dict[str, List[CFunctionDef]]]:
    by_name: Dict[str, List[CFunctionDef]] = {}
    by_file: Dict[str, List[CFunctionDef]] = {}
    for root, _, files in os.walk(c_proj):
        for filename in files:
            if not (filename.endswith(".c") or filename.endswith(".h")):
                continue
            path = os.path.join(root, filename)
            try:
                text = open(path, "r", encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            defs = scan_c_functions(text, path)
            by_file[path] = defs
            for d in defs:
                by_name.setdefault(d.name, []).append(d)
    return by_name, by_file


def collect_macro_names(c_proj: str) -> set:
    macros = set()
    define_re = re.compile(r"^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    for root, _, files in os.walk(c_proj):
        for filename in files:
            if not (filename.endswith(".c") or filename.endswith(".h")):
                continue
            path = os.path.join(root, filename)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        m = define_re.match(line)
                        if m:
                            macros.add(m.group(1))
            except OSError:
                continue
    return macros


def build_call_graph(defs_by_name: Dict[str, List[CFunctionDef]]) -> Dict[str, List[str]]:
    names = set(defs_by_name.keys())
    graph: Dict[str, List[str]] = {name: [] for name in names}
    for name, defs in defs_by_name.items():
        calls = set()
        for d in defs:
            for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", d.body):
                token = match.group(1)
                if token in names and token not in C_KEYWORDS:
                    calls.add(token)
        graph[name] = sorted(calls)
    return graph


def find_c_files_for_base(c_proj: str, base: str) -> List[str]:
    out = []
    for root, _, files in os.walk(c_proj):
        for filename in files:
            if os.path.splitext(filename)[0] != base:
                continue
            if filename.endswith(".c") or filename.endswith(".h"):
                out.append(os.path.join(root, filename))
    return out


def guess_prefixes(c_names: List[str], struct_name: str) -> List[str]:
    prefixes = []
    struct_low = struct_name.lower()
    for cname in c_names:
        idx = cname.lower().find(struct_low)
        if idx >= 0:
            prefixes.append(cname[:idx])
    if not prefixes:
        return []
    counts = {}
    for p in prefixes:
        counts[p] = counts.get(p, 0) + 1
    return sorted(counts, key=lambda p: (-counts[p], len(p)))


def match_free_fn(rust_name: str, candidates: List[str]) -> List[str]:
    if rust_name in candidates:
        return [rust_name]
    for c in candidates:
        if c.lower() == rust_name.lower():
            return [c]
    for c in candidates:
        if c.rstrip("_") == rust_name:
            return [c]
        if rust_name.rstrip("_") == c:
            return [c]
    rust_norm = normalize_name(rust_name)
    norm_matches = [c for c in candidates if normalize_name(c) == rust_norm]
    if len(norm_matches) == 1:
        return norm_matches
    if norm_matches:
        return norm_matches
    return []


def score_method_candidate(cname: str, struct_name: str, method: str, prefixes: List[str]) -> int:
    score = 0
    c_low = cname.lower()
    struct_low = struct_name.lower()
    method_low = method.lower()
    norm = normalize_name(cname)
    if normalize_name(struct_name + method) == norm:
        score += 10
    if normalize_name(method + struct_name) == norm:
        score += 8
    if c_low.startswith(struct_low):
        score += 5
    if c_low.endswith(method_low):
        score += 4
    for pref in prefixes:
        if cname.startswith(pref) and struct_low in c_low:
            score += 3
            if method_low in c_low:
                score += 2
            break
    try:
        if norm.index(struct_low) <= norm.index(method_low):
            score += 2
    except ValueError:
        pass
    return score


def match_method_fn(struct_name: str, method: str, candidates: List[str]) -> List[str]:
    direct = match_free_fn(method, candidates)
    if len(direct) == 1:
        return direct

    struct_norm = normalize_name(struct_name)
    method_norm = normalize_name(method)
    if method in ("from_char", "from_chars"):
        hits = []
        for c in candidates:
            cnorm = normalize_name(c)
            if struct_norm in cnorm and "char" in cnorm:
                if "from" in cnorm:
                    hits.append(c)
                    continue
                if cnorm.find("char") <= cnorm.find(struct_norm):
                    hits.append(c)
        if hits:
            return hits
    if method == "to_char":
        hits = []
        for c in candidates:
            cnorm = normalize_name(c)
            if struct_norm in cnorm and "char" in cnorm:
                if cnorm.find(struct_norm) <= cnorm.find("char"):
                    hits.append(c)
        if hits:
            return hits
    if method.startswith("to_") and method not in ("to_char",):
        alt_method = method[3:]
        alt_norm = normalize_name(alt_method)
        hits = []
        for c in candidates:
            cnorm = normalize_name(c)
            if struct_norm in cnorm and alt_norm in cnorm:
                hits.append(c)
        if hits:
            prefixes = guess_prefixes(candidates, struct_name)
            scored = [(score_method_candidate(c, struct_name, alt_method, prefixes), c) for c in hits]
            scored.sort(reverse=True)
            best_score = scored[0][0]
            best = [c for s, c in scored if s == best_score]
            return best
    if method.startswith("is_"):
        alt_method = method[3:]
        alt_norm = normalize_name(alt_method)
        hits = []
        for c in candidates:
            cnorm = normalize_name(c)
            if struct_norm in cnorm and alt_norm in cnorm:
                hits.append(c)
        if hits:
            prefixes = guess_prefixes(candidates, struct_name)
            scored = [(score_method_candidate(c, struct_name, alt_method, prefixes), c) for c in hits]
            scored.sort(reverse=True)
            best_score = scored[0][0]
            best = [c for s, c in scored if s == best_score]
            return best
    hits = []
    for c in candidates:
        cnorm = normalize_name(c)
        if struct_norm in cnorm and method_norm in cnorm:
            hits.append(c)
    if not hits and "_" in method:
        tokens = [normalize_name(t) for t in method.split("_") if t]
        for c in candidates:
            cnorm = normalize_name(c)
            if struct_norm in cnorm and all(t in cnorm for t in tokens):
                hits.append(c)
    if not hits:
        return []
    prefixes = guess_prefixes(candidates, struct_name)
    scored = [(score_method_candidate(c, struct_name, method, prefixes), c) for c in hits]
    scored.sort(reverse=True)
    best_score = scored[0][0]
    best = [c for s, c in scored if s == best_score]
    return best


def resolve_symbol(symbol: RustSymbol, candidates: List[str], override_map: Dict[str, List[str]]) -> Tuple[List[str], str]:
    if symbol.kind == "free":
        key = f"free::{symbol.name}"
    else:
        key = f"method::{symbol.impl_type}::{symbol.name}"
    if key in override_map:
        override = override_map[key]
        if override is None:
            return [], "skipped"
        return override, "override"
    if symbol.kind == "free":
        matched = match_free_fn(symbol.name, candidates)
    else:
        matched = match_method_fn(symbol.impl_type or "", symbol.name, candidates)
    if not matched and symbol.kind == "free" and symbol.file_base:
        for variant in (f"{symbol.file_base}_{symbol.name}", f"{symbol.file_base}{symbol.name}"):
            matched = match_free_fn(variant, candidates)
            if matched:
                break
    if not matched and symbol.kind == "method" and symbol.file_base:
        matched = match_free_fn(symbol.file_base, candidates)
    if len(matched) == 1:
        return matched, "auto"
    if len(matched) > 1:
        return matched, "ambiguous"
    if symbol.kind == "method" and symbol.name in ("new", "to_char"):
        return [], "skipped"
    return [], "missing"


def trim_file(path: str, keep_names: set) -> Tuple[int, int]:
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    defs = scan_c_functions(text, path)
    if not defs:
        return 0, 0
    remove_ranges = []
    for d in defs:
        if d.name not in keep_names:
            start = d.start
            comment_start = text.rfind("/*", 0, start)
            comment_end = text.rfind("*/", 0, start)
            if comment_start != -1 and (comment_end == -1 or comment_end < comment_start):
                start = comment_start
            end = d.end
            while end < len(text) and text[end].isspace():
                if text[end] == "\n":
                    end += 1
                    break
                end += 1
            remove_ranges.append((start, end))
    if not remove_ranges:
        return len(defs), 0
    remove_ranges.sort()
    out = []
    last = 0
    removed = 0
    for start, end in remove_ranges:
        out.append(text[last:start])
        last = end
        removed += 1
    out.append(text[last:])
    new_text = "".join(out)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_text)
    return len(defs), removed


def load_overrides(path: Optional[str]) -> Dict[str, Dict[str, List[str]]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def process_project(
    c_proj: str,
    rust_proj: str,
    out_proj: str,
    overrides: Dict[str, Dict[str, List[str]]],
) -> Dict:
    if os.path.exists(out_proj):
        shutil.rmtree(out_proj)
    shutil.copytree(c_proj, out_proj)

    rust_ifaces = []
    iface_root = os.path.join(rust_proj, "src", "interfaces")
    if os.path.isdir(iface_root):
        for root, _, files in os.walk(iface_root):
            for filename in files:
                if filename.endswith(".rs"):
                    rust_ifaces.append(os.path.join(root, filename))
    rust_symbols: List[RustSymbol] = []
    for iface in rust_ifaces:
        rust_symbols.extend(parse_rust_interface(iface))

    c_defs_by_name, c_defs_by_file = collect_c_functions(out_proj)
    macro_names = {m.lower() for m in collect_macro_names(out_proj)}
    all_c_names = sorted(c_defs_by_name.keys())

    base_to_candidates: Dict[str, List[str]] = {}
    for sym in rust_symbols:
        if sym.file_base not in base_to_candidates:
            files = find_c_files_for_base(out_proj, sym.file_base)
            c_names = set()
            for f in files:
                for d in c_defs_by_file.get(f, []):
                    c_names.add(d.name)
            base_to_candidates[sym.file_base] = sorted(c_names)

    override_map = overrides.get(os.path.basename(c_proj), {})

    mappings = []
    root_names = set()
    for sym in rust_symbols:
        candidates = base_to_candidates.get(sym.file_base, all_c_names)
        matched, status = resolve_symbol(sym, candidates, override_map)
        if status == "missing" and sym.kind == "free" and sym.name.lower() in macro_names:
            matched = [sym.name]
            status = "macro"
        mappings.append(
            {
                "symbol": {
                    "kind": sym.kind,
                    "name": sym.name,
                    "impl_type": sym.impl_type,
                    "file_base": sym.file_base,
                },
                "c_matches": matched,
                "status": status,
            }
        )
        for m in matched:
            root_names.add(m)

    call_graph = build_call_graph(c_defs_by_name)
    keep_names = set(root_names)
    changed = True
    while changed:
        changed = False
        for name in list(keep_names):
            for callee in call_graph.get(name, []):
                if callee not in keep_names:
                    keep_names.add(callee)
                    changed = True

    file_stats = []
    total_defs = 0
    total_removed = 0
    for path in sorted(c_defs_by_file.keys()):
        defs_count, removed = trim_file(path, keep_names)
        total_defs += defs_count
        total_removed += removed
        file_stats.append({"path": path, "defs": defs_count, "removed": removed})

    return {
        "c_project": os.path.basename(c_proj),
        "rust_project": os.path.basename(rust_proj),
        "rust_interfaces": rust_ifaces,
        "rust_symbols": [
            {
                "kind": s.kind,
                "name": s.name,
                "impl_type": s.impl_type,
                "file_base": s.file_base,
            }
            for s in rust_symbols
        ],
        "mappings": mappings,
        "root_c_functions": sorted(root_names),
        "kept_c_functions": sorted(keep_names),
        "total_function_defs": total_defs,
        "total_removed_defs": total_removed,
        "file_stats": file_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Trim C projects based on Rust interface signatures.")
    parser.add_argument("--c-root", required=True, help="C dataset root.")
    parser.add_argument("--rust-root", required=True, help="Rust dataset root.")
    parser.add_argument("--out-root", required=True, help="Output C dataset root.")
    parser.add_argument("--overrides", help="JSON file with manual mapping overrides.")
    parser.add_argument("--limit", type=int, help="Process only first N projects.")
    parser.add_argument("--report", required=True, help="Path to JSON report.")
    args = parser.parse_args()

    overrides = load_overrides(args.overrides)

    if os.path.exists(args.out_root):
        shutil.rmtree(args.out_root)
    os.makedirs(args.out_root)

    c_projects = [
        os.path.join(args.c_root, d)
        for d in sorted(os.listdir(args.c_root))
        if os.path.isdir(os.path.join(args.c_root, d))
    ]

    if args.limit:
        c_projects = c_projects[: args.limit]

    report = {
        "c_root": args.c_root,
        "rust_root": args.rust_root,
        "out_root": args.out_root,
        "projects": [],
        "missing_rust_projects": [],
    }

    for c_proj in c_projects:
        rust_name = c2rust_prj_name(os.path.basename(c_proj))
        rust_proj = os.path.join(args.rust_root, rust_name)
        if not os.path.isdir(rust_proj):
            report["missing_rust_projects"].append(os.path.basename(c_proj))
            continue
        out_proj = os.path.join(args.out_root, os.path.basename(c_proj))
        proj_report = process_project(c_proj, rust_proj, out_proj, overrides)
        report["projects"].append(proj_report)

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
