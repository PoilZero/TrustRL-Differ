#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import tempfile


def collect_include_dirs(root: str) -> list[str]:
    dirs = set()
    for dirpath, _, filenames in os.walk(root):
        if any(name.endswith(".h") for name in filenames):
            dirs.add(dirpath)
            parent = os.path.dirname(dirpath)
            if parent:
                dirs.add(parent)
    dirs.add(root)
    return sorted(dirs)


def list_c_files(root: str, exclude_dirs: set[str]) -> list[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        parts = {p.lower() for p in dirpath.split(os.sep)}
        if parts & exclude_dirs:
            continue
        for name in filenames:
            if name.endswith(".c"):
                lower = name.lower()
                if lower.startswith("test") or lower.endswith("_test.c") or lower.endswith("_tests.c"):
                    continue
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def compile_c_file(
    cc: str, path: str, include_dirs: list[str], out_dir: str, root: str
) -> tuple[bool, str]:
    rel = os.path.relpath(path, root)
    obj_name = rel.replace(os.sep, "_") + ".o"
    obj_path = os.path.join(out_dir, obj_name)
    cmd = [cc, "-c", path, "-o", obj_path, "-std=c11"]
    cmd.extend(["-D_GNU_SOURCE", "-D_XOPEN_SOURCE=700"])
    for inc in include_dirs:
        cmd.extend(["-I", inc])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = proc.returncode == 0
    msg = (proc.stderr or proc.stdout).strip()
    return ok, msg


def compile_project(root: str, cc: str, exclude_dirs: set[str]) -> dict:
    include_dirs = collect_include_dirs(root)
    c_files = list_c_files(root, exclude_dirs)
    results = []
    ok_count = 0
    fail_count = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        for cfile in c_files:
            ok, msg = compile_c_file(cc, cfile, include_dirs, tmpdir, root)
            results.append({"file": cfile, "ok": ok, "message": msg})
            if ok:
                ok_count += 1
            else:
                fail_count += 1
    return {
        "project": os.path.basename(root),
        "c_files": len(c_files),
        "ok": ok_count,
        "failed": fail_count,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile-check trimmed C projects.")
    parser.add_argument("--root", required=True, help="Trimmed C dataset root.")
    parser.add_argument("--cc", default="gcc", help="C compiler (default: gcc)")
    parser.add_argument("--limit", type=int, help="Process only first N projects.")
    parser.add_argument("--report", required=True, help="Write JSON report to this path.")
    args = parser.parse_args()

    projects = [
        os.path.join(args.root, d)
        for d in sorted(os.listdir(args.root))
        if os.path.isdir(os.path.join(args.root, d))
    ]
    if args.limit:
        projects = projects[: args.limit]

    report = {
        "root": args.root,
        "projects": [],
    }
    exclude_dirs = {"test", "tests", "example", "examples", "bench", "benchmark", "benchmarks", "build"}
    for proj in projects:
        report["projects"].append(compile_project(proj, args.cc, exclude_dirs))

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
