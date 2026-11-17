# Recent Wondful work: branch snapshot and evidence

Prepared 2026-11-15. User requested uploading recent changes, visual results and encountered problems; explicitly selected `fix/wondful-render-review`.

## Scope

- Use this existing development checkout, based on `4c5d0ff`; preserve the separate `car-render-style-transfer` checkout and its unrelated dirty work.
- Commit the current relevant addon, tests and local E2E-tool changes without introducing another implementation change.
- Add a portable review report, representative original/final PNGs, evidence hashes and current offline verification results.
- Record the user's rejection of paint V2 as the current visual verdict. Preserve the distinction between technical execution, single-image subjective review and actual user acceptance.
- Compare the official 1.2.4 workflow with the experimental source-edit path; restoration is not implemented in this snapshot.

## Acceptance checks

1. Inspect changed-file scope and run the repository's offline tests. Any failure must be reported honestly; this task does not silently change implementation to manufacture a passing snapshot.
2. Evidence images must be copied byte-for-byte, readable, relative-linked and hash-listed. Label new generation versus existing-image replay accurately.
3. Exclude credentials, caches, local `.blend` projects, provider request/base64 dumps, temporary configs and downloaded third-party photographs. Include links/credits for reference research.
4. Inspect the staged diff and file inventory, commit and push normally to `fix/wondful-render-review`; no force push, merge into main, formal release or new image calls.
5. Verify the remote branch SHA matches the committed local SHA, and the separate checkout and original evidence assets remain untouched.
