#!/usr/bin/env python3
import argparse
import json
import os
import sys


def c2rust_prj_name(c_name: str) -> str:
    project_name = c_name.replace("-", "_")
    if project_name and project_name[0].isdigit():
        project_name = "proj_" + project_name
    if "." in project_name:
        project_name = project_name.split(".")[0]
    return project_name


def normalize_name(name: str) -> str:
    return name.lower().replace("-", "_")


def list_interface_files(rust_proj: str, interfaces_subdir: str) -> list[str]:
    interfaces_root = os.path.join(rust_proj, interfaces_subdir)
    if not os.path.isdir(interfaces_root):
        return []
    out = []
    for root, _, files in os.walk(interfaces_root):
        for filename in files:
            if filename.endswith(".rs"):
                out.append(os.path.join(root, filename))
    return out


def collect_c_basenames(c_proj: str, c_exts: set[str]) -> set[str]:
    basenames = set()
    for root, _, files in os.walk(c_proj):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in c_exts:
                base = os.path.splitext(filename)[0]
                basenames.add(normalize_name(base))
    return basenames


def build_c_map(c_root: str) -> dict[str, list[dict]]:
    c_map: dict[str, list[dict]] = {}
    for entry in os.listdir(c_root):
        path = os.path.join(c_root, entry)
        if not os.path.isdir(path):
            continue
        rust_name = c2rust_prj_name(entry)
        c_map.setdefault(rust_name, []).append(
            {"name": entry, "path": path, "basenames": None}
        )
    return c_map


def ensure_basenames(c_info: dict, c_exts: set[str]) -> set[str]:
    if c_info["basenames"] is None:
        c_info["basenames"] = collect_c_basenames(c_info["path"], c_exts)
    return c_info["basenames"]


def check_pair(
    rust_root: str,
    c_root: str,
    interfaces_subdir: str,
    c_exts: set[str],
    ignore_interfaces: set[str],
) -> dict:
    report = {
        "rust_root": rust_root,
        "c_root": c_root,
        "rust_projects": 0,
        "c_projects": 0,
        "unmapped_rust_projects": [],
        "ambiguous_rust_projects": [],
        "interface_total": 0,
        "missing_interfaces": [],
    }

    if not os.path.isdir(rust_root):
        report["error"] = f"Rust root not found: {rust_root}"
        return report
    if not os.path.isdir(c_root):
        report["error"] = f"C root not found: {c_root}"
        return report

    c_map = build_c_map(c_root)
    report["c_projects"] = sum(len(v) for v in c_map.values())

    rust_projects = [
        os.path.join(rust_root, d)
        for d in os.listdir(rust_root)
        if os.path.isdir(os.path.join(rust_root, d))
    ]
    report["rust_projects"] = len(rust_projects)

    for rust_proj in sorted(rust_projects):
        rust_name = os.path.basename(rust_proj)
        c_candidates = c_map.get(rust_name, [])
        if not c_candidates:
            report["unmapped_rust_projects"].append(rust_name)
            continue
        if len(c_candidates) > 1:
            report["ambiguous_rust_projects"].append(
                {"rust_project": rust_name, "c_projects": [c["name"] for c in c_candidates]}
            )

        interface_files = list_interface_files(rust_proj, interfaces_subdir)
        for rs_path in interface_files:
            base = os.path.splitext(os.path.basename(rs_path))[0]
            base_norm = normalize_name(base)
            if base_norm in ignore_interfaces:
                continue
            report["interface_total"] += 1
            found = False
            for c_info in c_candidates:
                basenames = ensure_basenames(c_info, c_exts)
                if base_norm in basenames:
                    found = True
                    break
            if not found:
                report["missing_interfaces"].append(
                    {
                        "rust_project": rust_name,
                        "interface_file": rs_path,
                        "c_candidates": [c["name"] for c in c_candidates],
                    }
                )

    return report


def find_dataset_pairs(datasets_root: str) -> list[tuple[str, str]]:
    pairs = []
    if not os.path.isdir(datasets_root):
        return pairs
    entries = [d for d in os.listdir(datasets_root) if os.path.isdir(os.path.join(datasets_root, d))]
    rbench_dirs = [d for d in entries if d.startswith("RBench_")]
    for rdir in sorted(rbench_dirs):
        suffix = rdir[len("RBench_") :]
        cdir = f"CBench_{suffix}"
        rpath = os.path.join(datasets_root, rdir)
        cpath = os.path.join(datasets_root, cdir)
        if os.path.isdir(cpath):
            pairs.append((rpath, cpath))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether each Rust interface file has a corresponding C source/header file."
        )
    )
    parser.add_argument(
        "--datasets-root",
        default="datasets",
        help="Root directory containing RBench_* and CBench_* datasets (default: datasets).",
    )
    parser.add_argument(
        "--rust-root",
        help="Specific RBench_* directory to check (overrides auto-pairing).",
    )
    parser.add_argument(
        "--c-root",
        help="Specific CBench_* directory to check (overrides auto-pairing).",
    )
    parser.add_argument(
        "--interfaces-subdir",
        default="src/interfaces",
        help="Relative path to Rust signature files (default: src/interfaces).",
    )
    parser.add_argument(
        "--c-exts",
        default=".c,.h",
        help="Comma-separated C file extensions to consider (default: .c,.h).",
    )
    parser.add_argument(
        "--ignore",
        default="",
        help="Comma-separated Rust interface basenames to ignore (e.g., mod,lib).",
    )
    parser.add_argument(
        "--report",
        help="Write a JSON report to this path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print missing interface details to stdout.",
    )
    args = parser.parse_args()

    c_exts = {ext.strip().lower() for ext in args.c_exts.split(",") if ext.strip()}
    ignore_interfaces = {normalize_name(x) for x in args.ignore.split(",") if x.strip()}

    reports = []
    if args.rust_root and args.c_root:
        reports.append(
            check_pair(
                args.rust_root,
                args.c_root,
                args.interfaces_subdir,
                c_exts,
                ignore_interfaces,
            )
        )
    else:
        pairs = find_dataset_pairs(args.datasets_root)
        if not pairs:
            print("No dataset pairs found. Use --rust-root and --c-root.", file=sys.stderr)
            return 2
        for rust_root, c_root in pairs:
            reports.append(
                check_pair(
                    rust_root,
                    c_root,
                    args.interfaces_subdir,
                    c_exts,
                    ignore_interfaces,
                )
            )

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2, ensure_ascii=True)

    for report in reports:
        if "error" in report:
            print(f"[ERROR] {report['error']}")
            continue
        rust_root = report["rust_root"]
        c_root = report["c_root"]
        print(f"[PAIR] {rust_root} -> {c_root}")
        print(f"  Rust projects: {report['rust_projects']}")
        print(f"  C projects: {report['c_projects']}")
        print(f"  Unmapped Rust projects: {len(report['unmapped_rust_projects'])}")
        print(f"  Interface files checked: {report['interface_total']}")
        print(f"  Missing interface files: {len(report['missing_interfaces'])}")
        if report["ambiguous_rust_projects"]:
            print(f"  Ambiguous Rust projects: {len(report['ambiguous_rust_projects'])}")
        if args.verbose and report["missing_interfaces"]:
            for item in report["missing_interfaces"]:
                print(
                    f"    MISSING {item['interface_file']} (C candidates: {item['c_candidates']})"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
