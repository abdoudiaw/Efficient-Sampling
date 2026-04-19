# Release Checklist

## Before tagging

1. Run `python3 -m unittest discover -s tests`.
2. Run `bash -n code/run code/test_tableI.sh code/test_tableII.sh code/test_eos.sh code/test_md.sh test_md.sh`.
3. Confirm the README links, badges, and DOI references are still correct.
4. Review `git status` for generated artifacts under `results/`.
5. Update `CITATION.cff` and `pyproject.toml` if project metadata changed.

## Suggested GitHub release notes

- Summarize documentation and packaging updates.
- Call out any workflow-level compatibility changes.
- Mention whether scientific results or only repository infrastructure changed.
- Link the paper DOI, Zenodo dataset DOI, and Code Ocean capsule DOI.

## Current public references

- Paper: https://doi.org/10.1038/s42256-024-00839-1
- Data: https://doi.org/10.5281/zenodo.10908462
- Code Ocean capsule: https://doi.org/10.24433/CO.1152070.v1
