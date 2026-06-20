# Post-Release Main Sync: 0.1.0

This note records the merge-preparation state for the `release/pypi-0.1.0` branch after the
successful PyPI `0.1.0` release and post-release device hardening.

## Branch Relation

The branch relation check was run from `release/pypi-0.1.0`:

```bash
git fetch origin
git merge-base --is-ancestor origin/main HEAD && echo "release branch contains main"
git log --oneline origin/main..HEAD
```

Result:

```text
release branch contains main
```

The release branch contains `origin/main` and is ahead by release, publishing, post-release
documentation, and device-hardening commits:

```text
4e93a9f fix(device): harden single-device execution paths
c09eec0 docs(release): audit device acceleration support
0066fa8 docs(release): update post-pypi release notes
62e51b2 docs(release): document pypi auth recovery
c5c183a chore(release): prepare 0.1.0
653b9dc ci(release): omit testpypi environment
171cb77 ci(release): fix testpypi environment syntax
57d4fe7 ci(release): prepare testpypi publishing
565fa4a chore(release): validate local package build
33a8bdc chore(release): prepare pypi metadata
668449d docs(release): audit pypi packaging
```

## Release-Doc Paths

Release documents are already normalised under `docs/release/`. The root-level paths
`docs/local_build_validation.md` and `docs/testpypi_release_plan.md` were not present, so no file
move was needed.

## Version Policy

PyPI `0.1.0` is live and remains the immutable public release. The source tree is now bumped to
`0.1.1.dev0` for post-0.1.0 development. The device-hardening changes on this branch are therefore
tracked as future patch-release work rather than a modification of the published `0.1.0`
distribution.

## Merge Boundary

No merge into `main` was performed during this preparation pass. The branch is structurally ready
to merge as a whole because it contains `origin/main` and only release, metadata, documentation,
workflow, and post-release device-hardening commits.
