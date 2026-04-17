# Contributing to Awesome-Embodied-AI

Thanks for helping improve this repository.

## What to contribute

- Add a missing paper, dataset, simulator, benchmark, toolkit, or project page.
- Fix a broken link, duplicate entry, wrong category, or inconsistent year.
- Propose a new section when a topic has enough high-quality resources to justify one.

## Before you open a PR

1. Search the README to make sure the resource is not already listed.
2. Put the entry in the most relevant section.
3. Prefer official links:
   - `Paper Link`: arXiv, OpenReview, CVF open access, publisher, or project paper page.
   - `Project Link`: official code repo, dataset page, benchmark page, simulator page, or project website.
4. Use the format below.

## Entry format

```md
- [x] Resource Title [[Paper Link]](https://...) [[Project Link]](https://...) [2025]
```

Legend:

- `[x]`: public code, dataset, benchmark, simulator, or toolkit is available.
- `[ ]`: paper only, project page only, or no maintained public repo was found.

If there is no valid `Project Link`, omit it instead of leaving an empty placeholder.

## Quality bar

- Keep titles exactly aligned with the paper or project title.
- Use one entry per resource per section.
- Avoid deep GitHub links such as `blob/...`, `tree/...`, or `?tab=readme-ov-file` when the repo root is enough.
- Use the paper year, not the repo creation year.
- If a work spans multiple topics, duplicate it sparingly and only when it genuinely helps navigation.

## Pull request checklist

- I checked for duplicates.
- I verified the paper link.
- I verified the project/code/dataset link.
- I used the correct year.
- I ran `python scripts/check_readme.py`.

## Small fixes

For link fixes, typos, or duplicate cleanup, feel free to open a focused PR without extra discussion.

For bigger taxonomy changes, open an issue first so the structure stays coherent.
