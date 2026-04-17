#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


README = Path(__file__).resolve().parents[1] / "README.md"
ENTRY_RE = re.compile(r"^- \[[ x]\] ")
HEADING_RE = re.compile(r"^(##|###) (.+)$")
YEAR_RE = re.compile(r"\[[0-9]{4}\]\s*$")
PAPER_RE = re.compile(r"\[\[Paper Link\]\]\(([^)]+)\)")
PROJECT_RE = re.compile(r"\[\[Project Link\]\]\(([^)]+)\)")


def normalized_title(line: str) -> str:
    title = ENTRY_RE.sub("", line)
    title = PAPER_RE.sub("", title)
    title = PROJECT_RE.sub("", title)
    title = YEAR_RE.sub("", title)
    return " ".join(title.split())


def main() -> int:
    lines = README.read_text().splitlines()
    issues: list[str] = []
    section_stack: list[str] = []
    seen: dict[tuple[str, ...], dict[str, int]] = {}

    for lineno, line in enumerate(lines, start=1):
        heading = HEADING_RE.match(line)
        if heading:
            level = 2 if heading.group(1) == "##" else 3
            section_stack = section_stack[: level - 2]
            section_stack.append(heading.group(2))
            continue

        if line.startswith("- - "):
            issues.append(f"line {lineno}: legacy '- -' list prefix")

        if not ENTRY_RE.match(line):
            continue

        if section_stack and section_stack[0] == "Legend":
            continue

        if not YEAR_RE.search(line):
            issues.append(f"line {lineno}: missing trailing [YYYY] year")

        if "[[Project Link]]()" in line:
            issues.append(f"line {lineno}: empty project link placeholder")

        paper = PAPER_RE.search(line)
        project = PROJECT_RE.search(line)
        if paper and project and paper.group(1) == project.group(1):
            issues.append(f"line {lineno}: project link duplicates paper link")

        if project:
            url = project.group(1)
            suspicious_parts = ("?tab=readme-ov-file", "/blob/", "/tree/", "arxiv.org/abs/")
            if any(part in url for part in suspicious_parts):
                issues.append(f"line {lineno}: project link should point to a stable root URL: {url}")

        section = tuple(section_stack)
        title = normalized_title(line)
        title_map = seen.setdefault(section, {})
        if title in title_map:
            issues.append(
                f"line {lineno}: duplicate title in section '{' / '.join(section)}' "
                f"(first seen on line {title_map[title]})"
            )
        else:
            title_map[title] = lineno

    if issues:
        print("README quality check failed:", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 1

    print("README quality check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
