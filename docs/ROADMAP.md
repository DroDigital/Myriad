# Myriad build roadmap

The game is built in major steps, with one pull request per step. Each step leaves `main` building, with tests green and something new to play or see. The design these steps implement is [design/myriad-synthesis.md](design/myriad-synthesis.md); section numbers below (§) refer to it.

| # | Step | Status |
|---|---|---|
| 1 | Foundation and GPU swarm | **Done** |
| 2 | Edible world, mass and the Queen | **Done** |
| 3 | Forms, Build and pathfinding | **Done** |
| 4 | Threats and the Kitchen act (minimum-fun prototype) | **Done** |
| 5 | Build depth: castes, mutations, species, Codex | Next |
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

## Step 4: Threats and the Kitchen act (done)

- **Threats** (`src/threats.rs`) are CPU entities, each holding a GPU `Threat` slot (8 slots, passed in a uniform buffer):
  - Its body takes contact damage, summed per slot in `counters[ENEMY_HITS..]`.
  - Its zone (circle, cone or lane) slows and kills critters with a per-step chance, higher where they're packed, so **Spread is a defence**.
  - Every attack is telegraphed first on the overlay.
  - **Spray bottle**: a cone of mist. **Wiping cloth**: sweeps a lane across the swarm.
  - The Ring crushes enemies (3x contact damage) and the Spear 2x. A Comet knocks enemies back and interrupts them.
- **Response level**: rises with mass (one level per doubling above 100) and with time survived.
  - It paces spawns (up to 3 alive).
  - It counters habits: heavy Shield use brings sprays in pairs, heavy Spear use brings cloths.
  - A narrator line marks each new level.
- **The Cleaner** arrives at 800 mass or with half the kitchen eaten.
  - It has 2,000 hp and three phases: spray, then spray alternating with wipe, then a faster double spray.
  - It **enrages** over time (faster, and harder on the Queen), so ignoring it loses.
- **Run lifecycle**: title screen (Enter), playing, transition, won and lost screens, each with a summary (time, peak mass, % eaten).
  - Beating the Cleaner starts the **act transition**: the camera pulls back, and a GPU `merge` pass folds every other agent per cell into its neighbour while critters per agent double. Mass is kept (rounded in the player's favour).
- **The Queen** regenerates 3 hp/s after 4 s unhurt. The mop is gone.
- **Headless**:
  - `--threats`, `--boss-now`, `--passive` and `--bench` (32,768 critters for 600 steps).
  - The scripted player fights: it steers onto nearby enemies, raises the Shield when an attack is aimed at the Queen, and spears the boss.
  - In a boss rush with 1,500 critters, the fighting script wins (about 11 s) and the passive one loses (about 130 s).
  - A full act from 10 critters reached the Cleaner at about 140 s and won at about 165 s.
- `docs/playtest.md` is the §10 checklist for human playtests.

Deferred, or for playtesting to settle:
- **Balance.** The act is short (about 3 minutes against the design's 5), and the Cleaner falls fast to Spear and Ring. Tune with real players using the knobs listed in `playtest.md`.
- The Comet's knockback and the Bloom have no dedicated tests.
- The narrator is HUD text only, with no voice.
- Act 2 (the Garden) is just a banner.
- Enemies ignore walls while moving (they clamp to the world bounds).

## Step 5: Build depth

**Goal:** runs differ. What you hatch, the mutations you pick and the species you play change how the swarm fights and grows (§4.3, §4.5).

Scope:
1. **Castes** (§4.3): Workers, Soldiers, Spitters, Fliers, Nurses.
   - Use `Agent.caste`, which is already reserved. The Queen's **laying ratio** sets the mix of hatchlings (hatch passes pick a caste by ratio from a hash).
   - Per-caste multipliers live in a uniform table generated from Rust, as the kind table is: speed, eat rate, contact damage, hazard resistance, plus Nurse hatch-cost discounts.
   - Render each caste in its own colour; keep the colours fixed and colour-blind safe.
2. **Bloom meter and mutation cards**: eating fills a meter; each fill pauses the game and offers 3 cards.
   - Mutations live in a data file (RON or a simple text format) of about 20 to start, each modifying a caste, a Form, the Queen or the laying ratio, plus 3 **evolutions** that combine two owned mutations (e.g. Acid Mandibles + Spore Bloom → Corrosive Bloom).
   - Show cards with the bitmap font; keys 1–3 or the D-pad pick.
3. **Species**: four parameter sets (Ants, Locusts, Slime cells, Fireflies) chosen on the title screen. Each changes `SwarmTuning` and gives one signature trait: Ants make cheaper bridges, Locusts eat faster in short hops, Slime cells merge to toughen, Fireflies' Bloom detonates.
4. **Evolution Codex**: a meta-save (a JSON file in the platform data directory via `dirs`) unlocking species, Forms and mutations by milestones. It gives breadth, not power.
5. Tests: caste ratios hold in hatching (GPU), mutation effects apply, cards are offered deterministically from a seed, and the save round-trips.

Exit criteria:
- Two headless runs with different seeds and choices end measurably differently: caste mix, time to the Cleaner, and mass curve.
- Tests pass.
- Screenshots of the card pick, a caste-mixed swarm, and each species.

### Prompt to continue in a new session

```text
You are continuing development of Myriad, a Rust + wgpu game in this repository.
Read docs/ROADMAP.md first (conventions, including the 8-storage-buffer limit
on the steer pass, and what Steps 1-4 built), then the scope and exit criteria
for "Step 5: Build depth", then docs/design/myriad-synthesis.md §4.3 and §4.5.
Then skim src/sim/passes.wgsl, src/sim/mod.rs, src/run.rs, src/threats.rs,
src/forms.rs and src/hud.rs: the GPU owns the population and the material,
the CPU sends a SimFrame (including threats) and reads cumulative counters back
asynchronously, and the gameplay rules in Run are pure CPU code.

Implement Step 5 as scoped, in small tested increments: castes first, then
mutations and the Bloom meter, then species, then the Codex. Use Mesa lavapipe
if there is no GPU (apt-get install -y mesa-vulkan-drivers libudev-dev). Run
cargo fmt, cargo clippy --all-targets -- -D warnings and
MYRIAD_REQUIRE_GPU=1 cargo test. Check headless runs and look at screenshots
before finishing; save them to docs/media/step5-*.png. Update docs/ROADMAP.md:
mark Step 5 done, note anything deferred, and write the Step 6 continuation
prompt in the same style. Commit, push to your working branch, and open a PR
titled "Step 5: Build depth".
```

## Step 6: Presentation

The cosmetic micro-swarm layer that renders the honest million (§5). The macro look: depth of field, then tilt-shift as the scale grows. A density light grid for night acts. Audio: Kira plus a granular swarm synthesizer, with music layers at 10k, 100k and 1M. Settings and accessibility (§11).

## Step 7: Web demo

The Kitchen act built to WASM/WebGPU from the same code. Detect WebGPU and show a friendly fallback, keep the download under 25 MB, unlock audio on first input, and add a CI job that builds the web demo artifact.
