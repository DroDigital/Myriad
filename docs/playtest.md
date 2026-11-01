# Playtest checklist: minimum-fun prototype (§10)

Steps 1–4 make up the minimum-fun prototype. These are the §10 exit questions; answer them with people who haven't seen the game before. Write results in the table at the bottom.

## Setup

```bash
cargo run --release
```

- **Keyboard:** WASD or mouse to steer, Space to Gather, Shift to Spread, Z/X for Forms, Q/E to cycle Forms, B to Build, Enter to start, R to restart.
- **Gamepad:** stick, triggers, A/B for Forms, X to Build, bumpers to cycle Forms, Start.
- Don't explain anything beyond the controls card. Watch, and take notes.

## Questions

1. **Steering feel.** After 2 minutes, ask: "Does the swarm go where you want?" Score 1–5; most testers should give 4 or more. Note any moment they fight the controls: leash, drift, overshoot.
2. **One more run.** Count how many runs each tester plays *without being asked*. The target is 3 or more.
3. **The five-second clip.** Show a stranger 5 seconds of footage (the growing swarm, a Spear, the Cleaner). Ask "What's going on?" Pass if they say something like "you are the swarm" or "you're the horde".
4. **Fair Queen deaths.** After each loss, ask: "Did you see that coming?" Pass if testers say yes or "my fault" for most deaths. Watch for deaths to an attack they didn't see telegraphed.
5. **Forms read.** Do testers find the Spear and Shield without prompting, and do they understand what each did?
6. **Threats read.** Can testers point at the telegraph before a spray or wipe lands, even with 1,000+ critters on screen?
7. **Performance.** On Steam Deck–class hardware, run `cargo run --release -- --bench` and play a full act. Target 60 fps at 800p with the full swarm.

## Known tuning knobs

| Feel | Where |
|---|---|
| Swarm speed and pressure | `SwarmTuning` in `src/sim/mod.rs` |
| Form costs, durations, cooldowns | `FormKind::def` in `src/forms.rs` |
| Threat damage, lethality, health, pacing | `src/threats.rs` (`Enemy::zones`, `*_HEALTH`, `Response`) |
| When the Cleaner arrives | `RunConfig::boss_mass` in `src/run.rs` |
| Queen health and regeneration | `src/queen.rs` |

## Results

| Date | Tester | Steering (1–5) | Voluntary runs | Clip test | Fair deaths | Notes |
|---|---|---|---|---|---|---|
| | | | | | | |
