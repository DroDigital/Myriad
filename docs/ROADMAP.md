# Myriad build roadmap

The game is built in major steps, with one pull request per step. Each step leaves `main` building, with tests green and something new to play or see. The design these steps implement is [design/myriad-synthesis.md](design/myriad-synthesis.md); section numbers below (§) refer to it.

| # | Step | Status |
|---|---|---|
| 1 | Foundation and GPU swarm | **Done** |
| 2 | Edible world, mass and the Queen | **Done** |
| 3 | Forms, Build and pathfinding | **Done** |
| 4 | Threats and the Kitchen act (minimum-fun prototype) | Next |
| 5 | Build depth: castes, mutations, species, Codex | Planned |
| 6 | Presentation: the cosmetic million, macro look, audio | Planned |
| 7 | Web demo (WebGPU) | Planned |

Steps 1–4 make up the **minimum-fun prototype** from §10. Its exit criteria are the gate before Steps 5–7.

## Conventions every step follows

- Rust 2024 with wgpu 30 and winit 0.30. No game engine.
- GPU code is WGSL. The swarm passes live in `src/sim/passes.wgsl`, one `//! pass:<name>` section per pass. Each section is compiled with the material kind table (generated from Rust) and `src/sim/common.wgsl` prepended, and uses auto-derived bind group layouts. With auto layouts, every declared binding must actually be used, or bind group creation fails.
- The GPU owns the critter population. Agent passes dispatch over the full capacity and skip slots at or above `counters[LIVE_END]`. The CPU asks for hatchlings (`SimFrame::hatch`) and learns the outcome from cumulative counters in `SwarmSummary`.
- The default limit is **8 storage buffers per shader stage**, and the steer pass uses all 8. Before adding a binding to it, merge buffers (as `grid` merges counts and starts) or reuse spare fields (water cells keep their bridge count in `mat_amount`).
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
- **Headless**: a scripted player tours the food, waiting to arrive and leaving once eating dries up. From 10 critters it reaches about 3,160 in 3 minutes (52% of the kitchen eaten).

Deferred:
- **Pathfinding.** Critters steer in straight lines, so some get stuck behind obstacles (bread, cupboards) until they eat through. This is now the first item of Step 3.
- The Queen collides with walls but walks over uneaten solid food (the CPU doesn't track food amounts).
- The mop is a stand-in until Step 4's enemies.

## Step 3: Forms, Build and pathfinding (done)

- **Flow-field pathfinding**: a `flow` pass relaxes a path-distance field toward the focal point over the 1-unit grid, 6 sweeps per step on the GPU, from live material state.
  - Walls are impassable. Uneaten solid food is costly (`solid_cost`) but passable, so the swarm chews through when there is no way round. Open water is impassable unless Build is held.
  - Critters follow the field's downhill direction and slow down by path distance.
  - Result: 100% of critters stay in the main body over a 3-minute scripted run; stragglers no longer get stuck behind the bread.
- **Forms** (`src/forms.rs`, shapes in `form_pull` in `passes.wgsl`):
  - Two slots; the default loadout is Spear and Shield, and Q/E cycle each slot through all five.
  - Each Form costs a share of mass (from the bank first, then by sacrificing critters, counted as "spent", not "lost"). Each lasts a fixed time, then cools down.
  - **Spear**: a capsule along the aim. Each critter takes a slot along it, the swarm is carried forward, and critters in it eat 4x as fast. It bores about 6x deeper into cereal than plain swarming (tested).
  - **Shield**: a turtle shell around the Queen, sized to the swarm. Critters in it ignore hazards, and the Queen takes 15% damage. In the test it takes deaths from 500 to 0.
  - **Ring**: a closing circle where critters eat 3x as fast.
  - **Comet**: a dash.
  - **Bloom**: an outward burst where critters eat 2x as fast.
- **Build (Bridge)**: hold B.
  - A critter that bumps into open water anchors in it, and water cells count their anchored critters (stored in `mat_amount`) to become walkable.
  - Let go and the bridge drops: critters left in the water may drown.
  - The kitchen now has a water moat around the cheese.
- **Input**: Z/X fire the Form slots, Q/E cycle them, and B holds Build.
  - Gamepad (gilrs): stick steers, triggers Gather/Spread, A/B fire the Forms, X holds Build, and the bumpers cycle.
  - The mouse cursor or the stick aims.
- **HUD**: Form slots show their cost, cooldown or ACTIVE; the Build indicator shows when it's held.

Deferred:
- **Tower and Raft** need raised terrain (counter tops) and floating; Bridge is the only Build so far.
- Forms run for a fixed time rather than for as long as the button is held.
- Ring, Comet and Bloom have little to push against until Step 4's enemies.
- The Queen walks over water (the CPU doesn't know where bridges are).
- Gamepad input is untested on real hardware.

## Step 4: Threats and the Kitchen act (minimum-fun prototype)

**Goal:** something fights back, and the kitchen is a complete run. This is the last step of the §10 minimum-fun prototype.

Scope:
1. **Enemy framework** (CPU entities, like the mop, rendered on the overlay layer):
   - Each enemy has health, which critters wear down by contact. Report this as a per-enemy damage sum from the GPU: add an `enemy` buffer the steer pass atomically adds bites to, merged into an existing binding (the steer pass is at the 8-storage-buffer limit).
   - Telegraphs go on the high-contrast overlay layer. Replace the mop with real threats:
     - a **spray bottle** that telegraphs a cone, then mists it (kills critters, slows the rest);
     - a **wiping cloth** that sweeps a lane.
   - Ring now crushes enemies, and Comet knocks them back.
2. **Response level** (§4.3): a rising pressure tied to mass and time.
   - It spawns more and tougher threats.
   - It counters the player's dominant style (e.g. lots of Shield use brings sprays that come from two sides).
   - The narrator-style HUD line announces each change ("The kitchen has noticed you.").
3. **The Cleaner** boss closes the act. It combines spray and wipe, and Shield and Spread are its counters. It has multiple phases and a health bar.
4. **Run lifecycle**:
   - title / start, win (Cleaner defeated) and lose (Queen falls) screens, with a summary (time, peak mass, % eaten) and restart;
   - an **Act 1 → Act 2 transition** stub: the camera pulls back and critters per agent double (merge pairs of agents in a GPU pass, keeping mass); show "THE GARDEN" and end the prototype there.
5. **Prototype exit check (§10)**: write `docs/playtest.md` with a checklist for human testers:
   - steering feel;
   - 3+ voluntary runs;
   - the five-second clip test;
   - Queen deaths that feel fair;
   - performance on Steam Deck-class hardware.
   Add a `--bench` headless preset that reports ms/step at 32k agents.

Exit criteria:
- A scripted headless run can beat the Cleaner with Forms, and a passive run loses.
- Tests: enemy health is reduced by contact; the agent merge preserves mass; Response escalates with mass; the win and lose states.
- Screenshots of the spray telegraph, the Cleaner fight and the win screen.

### Prompt to continue in a new session

```text
You are continuing development of Myriad, a Rust + wgpu game in this repository.
Read docs/ROADMAP.md first (conventions, including the 8-storage-buffer limit
on the steer pass; what Steps 1-3 built; and the scope and exit criteria for
"Step 4: Threats and the Kitchen act"), then docs/design/myriad-synthesis.md
§4.3, §4.4 and §10. Then skim src/sim/passes.wgsl, src/sim/mod.rs, src/run.rs,
src/forms.rs, src/hazard.rs and src/session.rs: the GPU owns the population and
the material, the CPU sends a SimFrame and reads cumulative counters back
asynchronously, and the gameplay rules in Run are pure CPU code.

Implement Step 4 as scoped, in small tested increments: the enemy framework
first, then Response, then the Cleaner, then the run lifecycle. Use Mesa
lavapipe if there is no GPU (apt-get install -y mesa-vulkan-drivers
libudev-dev). Run cargo fmt, cargo clippy --all-targets -- -D warnings and
MYRIAD_REQUIRE_GPU=1 cargo test. Check headless runs and look at screenshots
before finishing; save them to docs/media/step4-*.png. Update docs/ROADMAP.md:
mark Step 4 done, note anything deferred, and write the Step 5 continuation
prompt in the same style. Commit, push to your working branch, and open a PR
titled "Step 4: Threats and the Kitchen act".
```

## Step 5: Build depth

Castes and the Queen's laying ratio. A Bloom meter with mutation cards, defined in data files (start with about 20, plus 3 evolutions). Four species parameter sets (Ants, Locusts, Slime cells, Fireflies). An Evolution Codex meta-save.

## Step 6: Presentation

The cosmetic micro-swarm layer that renders the honest million (§5). The macro look: depth of field, then tilt-shift as the scale grows. A density light grid for night acts. Audio: Kira plus a granular swarm synthesizer, with music layers at 10k, 100k and 1M. Settings and accessibility (§11).

## Step 7: Web demo

The Kitchen act built to WASM/WebGPU from the same code. Detect WebGPU and show a friendly fallback, keep the download under 25 MB, unlock audio on first input, and add a CI job that builds the web demo artifact.
