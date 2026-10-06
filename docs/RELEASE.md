# Releasing

Two paths — the automated one is preferred once the workflow is on `main`.

## Automated (preferred)

1. Commit everything that belongs in the release; push `main`.
2. Tag and push the tag — the `release` workflow builds on a clean macOS arm64
   runner and attaches `laya-rust-server-<tag>-macos-arm64.tar.gz` (+ `.sha256`)
   to a GitHub Release automatically:

   ```bash
   git tag -a v1.2.0 -m "v1.2.0"
   git push origin v1.2.0
   ```

3. Watch the run under the **Actions** tab; the release appears under
   **Releases** with the artifacts attached.

## Manual (fallback)

```bash
# build from the exact tagged state
git checkout v1.1.0
cd onnx-rust-server && cargo build --release --locked

# package
TAG=v1.1.0
PKG="laya-rust-server-${TAG}-macos-arm64"
mkdir -p "$PKG"
cp target/release/laya-rust-server "$PKG/"
cp ../onnx-server/node_modules/onnxruntime-node/bin/napi-v6/darwin/arm64/libonnxruntime*.dylib "$PKG/"
cp dist-files/RUN.md dist-files/ai.local.laya-warm.plist.example "$PKG/"
tar -czf "${PKG}.tar.gz" "$PKG"
shasum -a 256 "${PKG}.tar.gz" > "${PKG}.tar.gz.sha256"
```

Then: GitHub → Releases → **Draft a new release** → choose the tag → paste the
`CHANGELOG.md` section → drag & drop the `tar.gz` + `.sha256` → Publish.

## Integrity

Consumers verify with:

```bash
shasum -a 256 -c laya-rust-server-<tag>-macos-arm64.tar.gz.sha256
```

Note: the binaries are ad-hoc signed (no Developer ID), so macOS Gatekeeper may
scan them on first launch — expected; see docs/OPERATIONS.md for the AV notes.
