# TFMBot: local simulator and training candidate

Research date: 2026-10-06. Addresses [Investigate tfmbot as a local simulator and training starting point](https://github.com/finger-six/tm_bot/issues/11). Candidate pinned to [`b6574c95a5de624d8e979c9931fce2b3a9e01d4c`](https://github.com/rubberduckmaths/tfmbot/commit/b6574c95a5de624d8e979c9931fce2b3a9e01d4c), committed 2026-10-03T00:47:43Z. Source was retrieved through GitHub's API at that explicit revision; citations below use pinned files and line ranges.

## Recommendation and evidence boundary

**Keep tfmbot as a promising binary-runtime experiment, not the selected source-maintainable engine or a ready-made local bot/training stack.** It includes a real prebuilt WASM rules engine with seed, decision, legal-action and byte-snapshot interfaces. Those are useful simulator seams. Its supplied controller delegates the opponent to tfmbot.com; the public repository does not supply the C rules source, native bot/server implementation, weights or training pipeline. The requested five publisher maps are not established, and actor observations require explicit redaction. [Architecture][readme-architecture] [Worker API][worker-api] [Repository tree][tree]

This is source inspection only. No downloaded code was executed; no install, build, game rollout, remote bot call or benchmark was performed. The binary's internal rules, actual session size, clone correctness, RNG continuity and throughput remain unverified. Browser tests exist but were not run. Local RTX 4080 Super (16 GB VRAM) and 32 GB RAM are target hardware, not measured feasibility evidence. [Test description][readme-tests]

## What is public and what runs locally

| Component | Evidence and reuse boundary |
| --- | --- |
| Web UI and controller | Plain JavaScript, board/card rendering, human decision dialogs and replay recorder are public. README calls this the web client. [README][readme-architecture] |
| Rules/session runtime | C rules engine distributed as prebuilt `web/wasm/tfmweb.wasm` plus generated Emscripten glue. The worker calls it locally for state, legal choices and human answers. This is more than a remote-only client. [README][readme-architecture] [API calls][worker-api] [Answer dispatch][worker-answer] |
| Maintainable rules source | No `.c`/`.h` engine files or C build instructions in the pinned, non-truncated repository tree. README identifies the prebuilt engine without linking its source. A bounded check of the author's five public repositories found tfmbot and four scraper/card-list repositories, with no linked C-engine repository. This does not establish that source cannot be obtained elsewhere or from the author. [Tree][tree] [README][readme-architecture] [Public repository listing][author-repos] |
| TFMBot inference/search | Hosted. Worker posts the complete session bytes to `bot?op=step&sims=...` and restores returned bytes. `hint`, `sellhint` and `auto` also call the server. No public local bot implementation was found in the tree. [Server calls][worker-bot] [Hint/autoplay][worker-autoplay] |
| Evaluation | Development server forwards `/api/eval` as well as `/api/bot`; it implements neither locally. [Proxy][serve] |
| Weights, training and self-play learner | No model checkpoints, training scripts or optimizer/data-loader stack in the pinned tree. Search-based AI wording, `sims`, and a UI comment mentioning nets do not establish a reusable ML model or GPU pipeline. [Tree][tree] [README][readme-architecture] |
| Human-data collection starting point | Recorder captures full views, human answer payloads, decision kinds and terminal winner/tie/VP. Useful starting material for human warm-start data, requiring observation filtering and action-label normalization. This is a recorder, not a supplied trainer. [Recording][recorder] [Answers/results][record-payload] |

`npm start` serves static `web/` and proxies bot/eval to the remote service by default. Other `/api/` routes, including game logs and replays, return 204 and store nothing in the development server. Changing its `API` URL only redirects requests; it does not create a local bot. Accordingly, the supplied local app cannot finish ordinary play offline once it needs an opponent move. Local rules calls themselves do not intrinsically require that server. [Local server][serve] [Worker bot loop][worker-loop]

README/package identify GPL-3.0/GPL-3.0-only. Source acquisition and reproducible C builds remain practical prerequisites for independently maintaining rules or adding a missing map; the code license statement does not supply that missing source or audit asset rights. [License statement][readme-license] [Package][package]

## Simulator control, observations and branching

| Needed capability | Source-confirmed surface | Remaining check or adaptation |
| --- | --- | --- |
| Reset / first seat / map | `tw_new_game(seed, humanFirst, map)`; worker can accept explicit seed and `settings.humanFirst`. Omitted values use separate `Math.random()` calls. | Set all three explicitly. Same complete trajectory for identical seed/actions is untested; metadata also contains timestamps. [Reset][worker-reset] |
| Pending actor and phase | `tw_pending_kind`, `tw_pending_player`, `tw_human`; decision IDs include setup, draft, research, action, tile/trigger/keep/buy, Prelude and final greenery/end. | Scheduler must answer every decision kind and respect terminal `OVER`, not just main actions. [API][worker-api] [IDs][protocol] |
| Legal decisions / execution | `tw_legal_json`, `tw_pay_options`, action indices, setup/research/card/space/sell/keep/buy/trigger/payment answer functions. Glue additionally exports `tw_n_legal`, `tw_check_legal`, `tw_advance`. | Full signatures of extra exports and legal semantics are not supplied as source/API docs. Pending card/space choices are separate from main action lists; legality, invalid-answer atomicity and card/payment fidelity need tests. [API][worker-api] [Dispatch][worker-answer] [Glue][glue] |
| State and static tables | `tw_view_json`, `tw_static_json(map)`; replay schema describes resources/production/tags/tableau, tracks/tiles/milestones/awards, pending choices and outcomes. | JSON shape is useful, but **raw views are privileged**. Do not feed raw view, session bytes or both seats' hands to the policy. [API][worker-api] [Schema][view-schema] [Recording][recorder] |
| Serialization / restoration | `tw_session_ptr`, `tw_session_size`, `tw_session_restored`; snapshot copies session bytes with `HEAPU8.slice`, restore zeroes session memory, copies bytes and calls the restore hook. Layout hashes guard compatibility. | This is a concrete byte-snapshot mechanism, stronger evidence than a save-only JSON interface. Independent search branches, completeness at all nested decisions, RNG inclusion and repeatability are still untested. Glue exposes one session per module; an adapter could swap snapshots or create separate module instances. [Snapshot][worker-snapshot] [API][worker-api] |
| Chance handling | Seeded creation and session snapshots exist. Deck/discard count changes identify information revelation for UI undo. | No documented explicit chance-node sampler, RNG state accessor or hidden-state determinization API was found among the glue's named exports. This is an export-surface gap, not proof that binary internals lack RNG state. [Reset][worker-reset] [Undo logic][worker-after] [Glue][glue] |
| Offline two-policy self-play | Low-level answer functions and pending-player calls are separable from remote `botCall`. | Supplied worker refuses human answers on the bot seat, runs the remote bot automatically and gates displayed legal lists to the human seat. A separate controller must drive both seats and validate actor-independent legal queries. `autoplay` is remote, not offline self-play. [State/loop][worker-loop] [Answer guard][worker-answer] [Autoplay][worker-autoplay] |

Observation filtering already has reusable code: replay `redact()` hides opponent hand identities, setup deals, private draft/keep/buy/reveal information and selected private last-action fields. Setup additionally uses `App.mask()` to hide the opponent's earlier secret choice and derived standings. Replay seat switching changes the view's `human` field before applying both filters. These are candidate actor-observation transformations, not a leak-free guarantee: the schema retains VP breakdowns and other derived fields, so desired visibility and all information histories need an audit. Filters must run before generating training examples, including human recordings. [Redaction][redact] [Setup masking][mask] [Seat transformation][seat] [Schema][view-schema]

The recorder stores state views/diffs, not deterministic action re-simulation. It records human answer payloads but does not provide equivalent original answer payloads for the other seat. Therefore do not equate a replay file with a complete engine replay/search trajectory or clean labels for both policies. The development server also drops its upload endpoints; local collection/export needs adaptation. [Recorder design][recorder] [Payload/labels][record-payload] [Seat labels][seat] [Server][serve]

### Browser-free Node feasibility

**Plausible with a wrapper; not demonstrated as shipped.** The generated `.mjs` declares a Node environment flag, but its binary read functions are initialized only for web/worker environments; the non-web branch is empty. The ordinary worker loader supplies a URL, while the worker itself depends on browser `self.postMessage`. Simply importing the worker in Node is not a supplied headless simulator. [Glue][glue] [Worker loader][worker-loader]

The glue exposes `Module.instantiateWasm`: a prospective Node wrapper can supply local bytes through an instantiation callback, then reuse `cwrap`/`ccall` and memory access without the UI. That route requires an experiment to validate imports, initialization, randomness and every decision kind. Do not claim Node support or that merely supplying `wasmBinary` works: this generated loader has a local `wasmBinary` variable but no inspected argument-assignment path for it. Upstream's demonstrated browser-free *test runner* still launches Chromium and plays through the UI against the remote bot. [Glue][glue] [E2E runner][e2e]

## Target format and maps

Two-player UI/replay format is explicit (`NPLAYERS = 2`). Draft choice and research purchase dialogs exist. Setup requires two Prelude selections, and protocol/worker include multiple Prelude-resolution decisions. These establish exposed two-player/draft/Prelude paths, not measured completed-game correctness or proof of an exact **Prelude-only** card pool. The inspected new-game signature has only seed, first seat and map; no expansion/card-pool/draft toggles are exposed. Audit `tw_static_json` plus actual deals before accepting Corporate Era/base/original Prelude inclusion and absence of other sets. Text data is not executable card-rule evidence. [Two-player schema][view-schema] [Draft/research][draft] [Prelude setup][setup] [Reset][worker-reset]

| Requested publisher map | Exposed client evidence | Status for this project |
| --- | --- | --- |
| Tharsis | ID 0 | Selectable; publisher fidelity unverified. |
| Hellas | ID 1 | Selectable; publisher fidelity unverified. |
| Elysium | ID 2 | Selectable; publisher fidelity unverified. |
| Vastitas Borealis | ID 7, named **Vastitas Borealis Novus** by README | Selectable variant; name alone does not establish publisher map/bonus/milestone/award equivalence. |
| Amazonis Planitia | Not advertised or accepted by worker's map whitelist | Unsupported through inspected controller; `mapOk` silently changes unknown IDs to 0. Internal binary support remains unverified. |

Sources: [Advertised maps][readme-maps] [Worker whitelist][worker-maps] [Reset clamp][worker-reset]. Do not try undocumented IDs and infer success from getting a game: the wrapper substitutes Tharsis. No map layouts, parameter caps or complete card interactions were executed or decoded from binary in this investigation.

## Comparison and next decision

Compared with the [earlier source audit](https://github.com/finger-six/tm_bot/blob/ebbe499c543b0d63b24725b3b290480cb4adaf9b/docs/research/simulator-feasibility.md), tfmbot offers a smaller, concrete byte-snapshot seam and existing human UI/recording machinery. The audited `terraforming-mars` engine offers inspectable TypeScript rules, seeded in-process creation, structured player inputs and documented source adaptation points, but its arbitrary mid-decision JSON cloning is unproved and its implemented Amazonis is concretely incompatible with the requested publisher map. TFMBot does **not** eliminate that five-map problem: its exposed fourth map needs fidelity checking and its fifth is unestablished. [TFMBot snapshot][worker-snapshot] [Other engine construction][other-new] [Other engine snapshot][other-snapshot] [Publisher maps][publisher-maps] [Other Amazonis][other-amazonis]

Recommended bounded follow-up, before choosing a simulator:

1. Locate/obtain the C rules source, build provenance and card/map manifests from the author or linked material. If unavailable, decide explicitly whether dependence on an opaque pinned binary is acceptable. No author contact was made here.
2. If the binary path remains useful, build a strictly offline headless proof with explicit seed/first seat/map and two local policies. Verify complete games, all decision kinds, legal rejection, filtered actor views and byte round trips including draft/setup/payment/placement/trigger/final greenery. Compare two independent branches after subsequent random draws.
3. Compare `tw_static_json(7)` and actual rule outcomes with publisher Vastitas. Establish publisher Amazonis support from runtime/source evidence; if absent, maintaining that rules extension requires source access or selecting another candidate.
4. Measure reset/step/observation/clone latency and session bytes, games per second and sustained worker memory on the actual CPU/32 GB host. Measure policy inference/training VRAM separately on the 4080 Super, then together. No inspected timing field or server `sims` value establishes these local budgets.

For human warm-start, the recorder/redaction code is reusable design material. For subsequent local self-play, the WASM seam warrants one contained feasibility experiment only if rules/source/map requirements can be satisfied. Neither candidate is presently verified for the full five-publisher-map training goal; final engine selection remains the user's decision.

[readme-architecture]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/README.md#L3-L9
[readme-tests]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/README.md#L67-L75
[readme-maps]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/README.md#L35-L37
[readme-license]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/README.md#L77-L87
[tree]: https://api.github.com/repos/rubberduckmaths/tfmbot/git/trees/b6574c95a5de624d8e979c9931fce2b3a9e01d4c?recursive=1
[author-repos]: https://api.github.com/users/rubberduckmaths/repos?per_page=100
[package]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/package.json#L1-L19
[serve]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/tools/serve.mjs#L1-L48
[protocol]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/web/js/protocol.js#L1-L4
[worker-api]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/web/js/worker.js#L48-L82
[worker-maps]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/web/js/worker.js#L36-L44
[worker-snapshot]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/web/js/worker.js#L84-L94
[worker-bot]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/web/js/worker.js#L96-L120
[worker-loop]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/web/js/worker.js#L123-L169
[worker-after]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/web/js/worker.js#L171-L198
[worker-answer]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/web/js/worker.js#L235-L255
[worker-autoplay]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/web/js/worker.js#L273-L299
[worker-loader]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/web/js/worker.js#L305-L311
[worker-reset]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/web/js/worker.js#L313-L323
[glue]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/web/wasm/tfmweb.mjs#L1
[recorder]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/web/js/replay/replay.js#L3-L16
[record-payload]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/web/js/replay/replay.js#L176-L203
[redact]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/web/js/replay/replay_codec.js#L113-L143
[mask]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/web/js/app/app.js#L223-L241
[seat]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/web/js/replay/replay.js#L345-L356
[view-schema]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/web/js/replay/replay_codec.js#L192-L230
[draft]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/web/js/app/dialogs.js#L464-L483
[setup]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/web/js/app/dialogs.js#L355-L386
[e2e]: https://github.com/rubberduckmaths/tfmbot/blob/b6574c95a5de624d8e979c9931fce2b3a9e01d4c/tools/e2e.mjs#L1-L6
[other-new]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/Game.ts#L257-L332
[other-snapshot]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/Game.ts#L471-L532
[other-amazonis]: https://github.com/terraforming-mars/terraforming-mars/blob/f91e1e2574175c6affa2440a2c66a8fc00412e1e/src/server/boards/AmazonisBoard.ts#L7-L42
[publisher-maps]: https://fryxgames.se/wp-content/uploads/2024/09/TM_AV_WRAP_ENG.pdf