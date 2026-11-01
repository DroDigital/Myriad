# Myriad build roadmap

The game is built in major steps, with one pull request per step. Each step leaves `main` building, with tests green and something new to play or see. The design these steps implement is [design/myriad-synthesis.md](design/myriad-synthesis.md); section numbers below (§) refer to it.

| # | Step | Status |
|---|---|---|
| 1 | Foundation and GPU swarm | **Done** |
| 2 | Edible world, mass and the Queen | **Done** |
| 3 | Forms, Build and pathfinding | Next |
| 4 | Threats and the Kitchen act (minimum-fun prototype) | Planned |
| 5 | Build depth: castes, mutations, species, Codex | Planned |
| 6 | Presentation: the cosmetic million, macro look, audio | Planned |
| 7 | Web demo (WebGPU) | Planned |

Steps 1–4 make up the **minimum-fun prototype** from §10. Its exit criteria are the gate before Steps 5–7.

## Conventions every step follows

- Rust 2024 with wgpu 30 and winit 0.30. No game engine.
- GPU code is WGSL. The swarm passes live in `src/sim/passes.wgsl`, one `//! pass:<name>` section per pass. Each section is compiled with the material kind table (generated from Rust) and `src/sim/common.wgsl` prepended, and uses auto-derived bind group layouts. With auto layouts, every declared binding must actually be used, or bind group creation fails.
- The GPU owns the critter population. Agent passes dispatch over the full capacity and skip slots at or above `counters[LIVE_END]`. The CPU asks for hatchlings (`SimFrame::hatch`) and learns the outcome from cumulative counters in `SwarmSummary`.
- Gameplay rules live in `Run` (`src/run.rs`), pure CPU code that is unit-testable without a GPU. `Session` connects `Run` to the swarm, material and renderer.
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

## Step 2: Edible world, mass and the Queen (done)

- **Material grid** (`sim/material.rs`): 0.5-unit cells, each with a kind and a fixed-point food amount. Kinds are floor, wall, crumbs, bread, cereal, jam and cheese; each has flags (blocked, solid-while-food, edible) and an eat rate. The kind table is generated into WGSL from Rust.
- **Level files** (`assets/levels/kitchen.txt`, `src/level.rs`): an ASCII tile map listing the Queen start, the mop route and food clusters.
- **Movement and eating**: critters collide and slide along walls and uneaten solids. They eat food underfoot and gnaw solids they push into, using a bounded compare-exchange loop so food eaten always equals food removed (tested). A **forage** force draws them onto nearby food.
- **GPU-owned population**: a `counters` buffer tracks the live range plus cumulative hatched, eaten and died counts. The `hatch_revive` pass reuses dead slots, then `hatch_append` extends the range.
- **Feedback**: the reduce pass reports the *main body* centroid (critters near the focal point), the alive count and the Queen's guard count. Stragglers can't drag the Queen or the camera away.
- **Mass economy** (`colony.rs`): one full cell of food = 1 critter. Hatching is metered against in-flight requests and works from cumulative counters, so late feedback never double-counts.
- **The Queen** (`queen.rs`) follows the main body at 60% speed. Hatchlings appear around her, and nearby critters cut the damage she takes. At zero health she gets a 3-second last stand; then the run is lost and **R** restarts it.
- **The mop** (`hazard.rs`) is a placeholder hazard that patrols a route, wipes out critters and hurts the Queen.
- **Rendering**: smooth food blobs that shrink as they're eaten, an overlay layer for the Queen (beacon and health ring) and the mop, and a HUD from a built-in 5x7 bitmap font.
- **Headless**: a scripted player tours the food, waiting to arrive and leaving once eating dries up. From 10 critters it reaches about 365 in 2 minutes and about 1,570 in 3 (26% of the kitchen eaten).

Deferred:
- **Pathfinding.** Critters steer in straight lines, so some get stuck behind obstacles (bread, cupboards) until they eat through. This is now the first item of Step 3.
- The Queen collides with walls but walks over uneaten solid food (the CPU doesn't track food amounts).
- The mop is a stand-in until Step 4's enemies.

## Step 3: Forms, Build and pathfinding

**Goal:** the swarm's shape becomes the weapon and the tool (§4.3, Forms), and the swarm finds its way around the kitchen.

Scope:
1. **Flow-field pathfinding.** A GPU or CPU breadth-first search from the focal point over the material grid's walkable cells (coarse, about 1 unit, refreshed every few frames). Critters seek along the field's gradient instead of in a straight line. Treat uneaten solids as blocked but "eatable": a high cost rather than infinite, so a swarm with no way around chews through. Stragglers behind the bread should rejoin.
2. **Combat Forms** as signed-distance-field targets in the steer pass. Critters inside a Form get pulled to its shape and take on its property:
   - **Spear**: a line toward the cursor or heading; pierces solid food and armour fast.
   - **Shield**: a dome; blocks hazard damage (critters in the shell survive the mop).
   - **Ring**: encircles and crushes.
   - **Comet**: a dash.
   - **Bloom**: bursts outward.
   - Two Form slots, each Form with a mass cost and cooldown, drawn as rings around the swarm.
3. **Build Forms**: Bridge (across gaps or water), Tower (up counters) and Raft (on spills), all on one context button. Critters snap into lattice slots; a structure's load capacity scales with how many critters fill it. Add a sink gap and a counter top to the kitchen so Build matters (a new material kind for "drop" or water, and a raised area).
4. **Input**: gamepad via gilrs (stick, triggers for Gather/Spread, face buttons for Forms and Build), plus keyboard bindings for both Form slots and Build.
5. **HUD**: Form slots with cooldown rings.

Exit criteria:
- Headless: stragglers rejoin (main-body share of critters stays above 95% across a scripted tour); a scripted Spear eats through cereal faster than plain swarming; Shield cuts mop deaths by at least half.
- Tests: the flow field routes around a U-shaped wall; Form costs and cooldowns; a bridge spans a gap and critters cross it.
- Screenshots of each Form.

### Prompt to continue in a new session

```text
You are continuing development of Myriad, a Rust + wgpu game in this repository.
Read docs/ROADMAP.md first (conventions, what Steps 1-2 built, and the scope and
exit criteria for "Step 3: Forms, Build and pathfinding"), then
docs/design/myriad-synthesis.md §4.3. Then skim src/sim/ (mod.rs, passes.wgsl,
material.rs, feedback.rs), src/run.rs and src/session.rs to learn the patterns:
the GPU owns the population, the CPU sends a SimFrame and reads cumulative
counters back asynchronously, and the gameplay rules in Run are pure CPU code.

Implement Step 3 as scoped, in small tested increments, starting with
pathfinding. Use Mesa lavapipe if there is no GPU
(apt-get install -y mesa-vulkan-drivers). Run cargo fmt,
cargo clippy --all-targets -- -D warnings and MYRIAD_REQUIRE_GPU=1 cargo test.
Check the headless run (cargo run -- --headless --frames 7200 --screenshot
shot.png) and look at the screenshots before finishing. Save screenshots to
docs/media/step3-*.png. Update docs/ROADMAP.md: mark Step 3 done, note anything
deferred, and write the Step 4 continuation prompt in the same style. Commit,
push to your working branch, and open a PR titled
"Step 3: Forms, Build and pathfinding".
```

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
