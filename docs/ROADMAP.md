# Myriad build roadmap

The game is built in major steps, with one pull request per step. Each step leaves `main` building, with tests green and something new to play or see. The design these steps implement is [design/myriad-synthesis.md](design/myriad-synthesis.md); section numbers below (§) refer to it.

| # | Step | Status |
|---|---|---|
| 1 | Foundation and GPU swarm | **Done** |
| 2 | Edible world, mass and the Queen | Next |
| 3 | Forms and Build | Planned |
| 4 | Threats and the Kitchen act (minimum-fun prototype) | Planned |
| 5 | Build depth: castes, mutations, species, Codex | Planned |
| 6 | Presentation: the cosmetic million, macro look, audio | Planned |
| 7 | Web demo (WebGPU) | Planned |

Steps 1–4 make up the **minimum-fun prototype** from §10. Its exit criteria are the gate before Steps 5–7.

## Conventions every step follows

- Rust 2024 with wgpu 30 and winit 0.30. No game engine.
- GPU code is WGSL. The swarm passes live in `src/sim/passes.wgsl`, one `//! pass:<name>` section per pass. Each section is compiled with `src/sim/common.wgsl` prepended and uses auto-derived bind group layouts.
- Every struct shared between Rust and WGSL has a size test (see `gpu_struct_layouts_match_wgsl`).
- Gameplay never blocks on the GPU. Per-frame information comes back through `sim::feedback` (async, one or two frames late). Blocking readbacks (`read_agents`, `read_grid`) are for tests and tools only.
- `Game` holds no windowing code. Headless runs and tests drive the same `Session` that the window app does.
- Before opening a PR, run `cargo fmt`, `cargo clippy --all-targets -- -D warnings` and `MYRIAD_REQUIRE_GPU=1 cargo test`. Lavapipe is fine (`apt install mesa-vulkan-drivers`). Also save a headless screenshot to `docs/media/stepN-*.png` and look at it.
- At the end of a step, update this file: mark the step done, note anything deferred, and refresh the next step's prompt.

---

## Step 1: Foundation and GPU swarm (done)

- A Cargo crate with a library and a binary, plus a CLI (`--headless`, `--agents`, `--screenshot`…).
- wgpu device setup with a windowed surface or headless.
- Swarm sim on the GPU, up to 32,768 agents. Each step runs these passes:
  - **count**, **scan** and **scatter** build a spatial hash: per-cell counters, then a prefix sum, then an index list sorted by cell.
  - **steer** does flocking: separation, crowd pressure, alignment, cohesion and seek. It uses an unbiased per-cell neighbour budget, so the swarm never piles up on grid lines.
  - **reduce** writes per-workgroup sums, read back asynchronously so the CPU learns the swarm's centre without stalling.
- Gather/Spread, a focal point leashed to the swarm, and a camera that frames the swarm and pulls back as it grows (`zoom_for_count`).
- Renderer: a kitchen-tile floor, a focal ring, and instanced ant silhouettes that fall back to dots when far away.
- Tests: unit tests, and GPU integration tests covering the hash, flow, Gather/Spread, feedback, capacity and capture. CI runs them all on lavapipe.

Deferred to later steps: gamepad (Step 3), on-screen HUD text (Step 2), and per-act critters-per-agent merging (Step 4).

## Step 2: Edible world, mass and the Queen

**Goal:** the core loop of eating to grow. Critters devour the floor's contents, eaten matter becomes mass, mass hatches new critters at the Queen, and losing the Queen ends the run.

Scope:
1. **Material grid (§6, 2.5D).** A GPU grid at about cell resolution. Each cell holds material type, hardness, amount remaining and height. Blocked cells (walls, cupboard bases) stop critters and make the swarm flow around them. Start with a hand-authored kitchen layout in a data file: crumbs, spills, a cereal box, a bread loaf, table legs as obstacles.
2. **Eating.** Critters touching edible cells erode them with atomic writes. A per-step eat total is reduced on the GPU and returned through `sim::feedback`, extending `SwarmSummary` with an `eaten` amount. Render the material on the floor so it visibly disappears.
3. **Mass economy (§4.3).** Mass is one number: health, score and (later) Form currency. Eaten matter adds mass. Mass beyond the current critter count queues hatches.
4. **The Queen.** A larger, distinct agent. She follows the swarm's centre with lag, and **new critters hatch at her position**, not at the front. She has her own health, a beacon and an outline that are always visible, and a short last-stand window before the run ends. Use a temporary hazard to test her death, such as a slow-moving damage zone (a "mop").
5. **Critter loss.** Agents can die. Dead slots are reused by hatching, and dead critters are subtracted from mass.
6. **Minimal HUD.** A mass/critter counter and Queen health, drawn with a tiny bitmap font. No UI library.
7. Remove the `H` test hatch key, or put it behind `--dev`.

Exit criteria:
- Headless: a scripted run eats a measurable share of the kitchen's material and grows from 10 to at least 500 critters.
- Tests: material erosion conserves mass (eaten = removed); hatching happens at the Queen; Queen death ends the run; the swarm flows around blocked cells and never into them.
- A screenshot showing a partly eaten floor, the Queen and the HUD.

### Prompt to continue in a new session

```text
You are continuing development of Myriad, a Rust + wgpu game in this repository.
Read docs/ROADMAP.md first (conventions, and the scope and exit criteria for
"Step 2: Edible world, mass and the Queen"), then docs/design/myriad-synthesis.md
§4.3 and §6, then skim src/sim/ (mod.rs, passes.wgsl, feedback.rs),
src/session.rs and src/game.rs to learn the patterns that Step 1 set up.

Implement Step 2 exactly as scoped in the roadmap, in small, tested increments.
Use Mesa lavapipe for GPU work if there is no GPU
(apt-get install -y mesa-vulkan-drivers). Run cargo fmt,
cargo clippy --all-targets -- -D warnings and MYRIAD_REQUIRE_GPU=1 cargo test.
Render a headless screenshot to docs/media/step2-*.png and look at it before
finishing. Update docs/ROADMAP.md: mark Step 2 done, note anything deferred,
and write the Step 3 continuation prompt in the same style as this one.
Commit, push to your working branch, and open a PR titled
"Step 2: Edible world, mass and the Queen".
```

## Step 3: Forms and Build

Combat Forms (Spear, Shield, Ring, Comet, Bloom) are signed-distance-field targets in the steer pass, and each has an effect: pierce, block, crush, dash, burst. Build Forms (Bridge, Tower, Raft) use slot-snapped lattice structures, and their load capacity depends on how many critters fill the slots. Forms cost mass and have cooldowns, shown as rings around the swarm (§4.6). Two Form slots plus a context-sensitive Build button. Gamepad support through gilrs. The kitchen gets gaps and spills that need Bridge or Raft, and counters that need Tower.

## Step 4: Threats and the Kitchen act (minimum-fun prototype)

- An enemy framework with telegraphs on a separate high-contrast layer, plus damage and death.
- A Response level that rises with mass and counters your dominant play style (§4.3).
- **The Cleaner** boss (spray cone and wiping cloth), countered by Shield and Spread.
- A run lifecycle (start, win, lose, restart), the Act 1 → Act 2 transition, and critters-per-agent merging during the camera pull-back (§5).
- Then run the §10 prototype exit check: steering feel, three or more voluntary runs, the five-second clip test, Queen deaths that feel fair, and Steam Deck-class performance.

## Step 5: Build depth

Castes and the Queen's laying ratio. A Bloom meter with mutation cards, defined in data files (start with about 20, plus 3 evolutions). Four species parameter sets (Ants, Locusts, Slime cells, Fireflies). An Evolution Codex meta-save.

## Step 6: Presentation

The cosmetic micro-swarm layer that renders the honest million (§5). The macro look: depth of field, then tilt-shift as the scale grows. A density light grid for night acts. Audio: Kira plus a granular swarm synthesizer, with music layers at 10k, 100k and 1M. Settings and accessibility (§11).

## Step 7: Web demo

The Kitchen act built to WASM/WebGPU from the same code. Detect WebGPU and show a friendly fallback, keep the download under 25 MB, unlock audio on first input, and add a CI job that builds the web demo artifact.
