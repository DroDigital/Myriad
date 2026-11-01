# Myriad

A survivors-like turned inside out: you are the horde. Hatch from ten critters on a kitchen floor, shape the swarm, guard your Queen, and eat your way to a million.

- **Design:** [docs/design/myriad-synthesis.md](docs/design/myriad-synthesis.md), the combined concept.
- **Build plan:** [docs/ROADMAP.md](docs/ROADMAP.md), the major steps, their status, and prompts for continuing in a new session.

![Three minutes in: the swarm devouring a jam spill](docs/media/step2-kitchen-3min.png)

## Run it

You need Rust 1.88 or newer, plus a GPU with Vulkan, Metal or DirectX 12.

```bash
cargo run --release
```

| Input | Action |
|---|---|
| WASD / arrow keys | Steer the swarm's focal point |
| Left mouse (hold) | Steer toward the cursor |
| Space | Gather (tighten) |
| Shift | Spread (loosen) |
| R | Restart |
| Esc | Quit |

Eat food to gain mass, and new critters hatch around the Queen. Keep her guarded: if she falls, the run ends. Watch for the mop.

Options: `--agents N`, `--capacity N`, `--seed N`, `--no-hazard` and `--dev` (H grants +100 mass). Run `cargo run -- --help` for the full list.

## Headless runs

Headless runs need no window. They're for benchmarks, screenshots and CI: a scripted player tours the kitchen's food, and the run prints timing, mass and swarm stats.

```bash
cargo run --release -- --headless --frames 7200 --screenshot shot.png
```

Add `--agents 20000` to benchmark a big swarm, or `--hazard` to include the mop.

On machines without a GPU, Mesa's software Vulkan driver works (`apt install mesa-vulkan-drivers`).

## Tests

```bash
cargo test
```

GPU tests skip themselves when no adapter is found. Set `MYRIAD_REQUIRE_GPU=1` (CI does) to make that a failure instead.

## Code map

| Path | What it is |
|---|---|
| `src/sim/` | GPU swarm: compute passes (`passes.wgsl`), material grid, async feedback, stats |
| `src/render/` | Floor and food, instanced critters, Queen and hazard overlay, bitmap-font HUD |
| `src/level.rs`, `assets/levels/` | ASCII level maps |
| `src/run.rs` | One run's rules: Queen, mop, mass economy, win/lose (pure CPU) |
| `src/game.rs` | Focal point, Gather/Spread, leash, camera targeting |
| `src/colony.rs`, `src/queen.rs`, `src/hazard.rs` | Mass economy, the Queen, the mop |
| `src/hud.rs` | HUD layout |
| `src/session.rs` | Fixed-timestep loop tying the GPU and the rules together |
| `src/app.rs` | winit window, input mapping, presentation |
| `src/headless.rs` | Windowless runs and PNG capture |
| `tests/gpu_swarm.rs` | GPU integration tests |
