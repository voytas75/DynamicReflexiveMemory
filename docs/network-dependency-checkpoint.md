# Network dependency qualification checkpoint

Status: local qualification; publication pending independent review and separate approval.

- Scope: uv.lock, CHANGELOG.md, and this checkpoint only; no staging, commit or push.
- Locked upgrades: aiohttp 3.14.3 → 3.14.4; urllib3 2.7.0 → 2.8.0; mandatory yarl 1.24.5 → 1.25.1.
- Results: 91 passed; 3 real-store integration tests deselected; 86.18% coverage (serial local regression, not a DRM CI gate).
- Existing frozen all-extras installation, pip check, Ruff lint/format and Pyright receipts passed.
- CTM xdist guard selftest: 5 passed; controller and all 4 automatic workers loaded the artifact guard through PYTHONPATH. No resource cap or installed .pth change was imposed.
- CTM exact suite: zero observed calls to patched network/provider/store methods; Python guards are not an OS sandbox and do not comprehensively cover native code, arbitrary subprocesses or other APIs.
- DRM exclusions: test_redis_working_memory_ttl_expiry, test_redis_store_recovers_after_restart, test_semantic_roundtrip_with_real_chroma. Historical DRM guards observed blocked DNS attempts; no zero-attempt claim for DRM.
- No live-provider/store acceptance or exploit-prevention claim. Existing unpatched Chroma risk decisions remain unchanged.
- Source, manifests, Python support, CI, security settings, provider/runtime configuration and roadmap are unchanged; no release-version bump is required for this unreleased lock-only slice.
- Earlier launcher failure receipts were overwritten; their historical trace is unavailable.
- Evidence: external drm-ctm-network and drm-ctm-network-closeout review packets; publication requires exact-byte independent review, then separate staging/commit/push authorization and remote verification.
