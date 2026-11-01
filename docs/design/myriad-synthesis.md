# Myriad: Combined Concept

**Status:** Draft for team review
**Sources:**
- **Concept 3**: the short brief. The swarm grows "from ten critters toward a million" and eats its way from a kitchen floor to a city skyline.
- **Concept 4**: the 15-page PDF. A glowing swarm devours colossal machines from the inside.

---

## 1. Summary

Both concepts start from the same idea. Each is a survivors-like where you play the horde. Mass works as health. A GPU simulates the swarm, and the game sells as a $9.99 premium title on Steam. They are strong in different places:

| | Concept 3 | Concept 4 |
|---|---|---|
| **Better at** | Fantasy, hooks, originality and marketing moments | Game systems, tech choices, scope discipline and validation |
| **Weaker at** | Technical scope, production risk and validation gates | Originality, broad appeal, and the size of the growth fantasy |

**Recommendation:** Take Concept 3's world, scale curve, Queen and marketing hooks. Build them on Concept 4's systems (Forms, castes, mass economy), its stack (Rust + wgpu with a web demo) and its staged validation plan. Where Concept 3 overreaches, cut it back: 1M simulated agents, a 3D voxel city, raw Vulkan, and online versus at launch.

The union of two proposals is always bigger than either one. Section 9 lists what this version deliberately does **not** take.

---

## 2. Review of each proposal

### Concept 3: Kitchen to Skyline

**Strengths**
- **A scale curve anyone can read.** Kitchen, garden, street, town, city. This is Katamari's real hook: you see your growth against familiar objects. No lore is needed to explain it.
- **The Queen.** One vulnerable unit turns the blob into a formation. Players defend by shaping the swarm. It is the best single mechanic in either proposal.
- **Living structures.** Bridges, towers and rafts make the swarm feel like a real colony. They also make strong clips.
- **An adaptive opponent.** "Lean on fire resistance and it brings water cannons." This answers the genre's "solved build" problem.
- **Marketing hooks.** A counter ticking past 1,000,000 as a skyscraper falls. A daily seeded city ranked on time-to-a-million. A comic narrator built for clips.
- **Tone and look.** Macro photography easing into tilt-shift is warm and distinctive, and it fits the scale theme. The cartoon-macabre treatment (shell-flake confetti, no gore) reaches a wide audience.

**Weaknesses**
- **Tech scope does not fit 3–5 people.** It asks for 1M simulated agents, position-based-dynamics structures, a 3D sparse voxel city with connectivity checks, Jolt debris, raw Vulkan with ports later, and host-authoritative versus netcode.
- **Raw Vulkan rules out a browser demo** and pushes macOS to "later".
- **Online versus at launch** needs a player base that a $9.99 indie title may not have. Empty lobbies hurt reviews.
- **Too much content.** About 40 mutations each for 6 species is around 240 mutations to design and balance.
- **No player-skill layer** beyond structures and Queen positioning. Combat stays mostly passive contact.
- **No validation gates** apart from one performance target.

### Concept 4: Colossus Interior

**Strengths**
- **Forms.** The shape of the swarm is the weapon: Spear, Shield, Ring and others. This adds timing and skill to an auto-fire genre. Each shape is cheap to build as a distance field in the swarm's compute shader.
- **Castes.** The build is the mix of mote types, not a set of weapon slots. Brood chambers set the hatch ratio.
- **Mass as the only resource.** One number is your health, your damage, your score and what Forms cost. Every decision becomes "how much of yourself do you spend?"
- **The Alarm level.** Pressure rises as you grow, so power and threat scale together.
- **The tech stack.** Rust + wgpu gives one codebase for Vulkan, D3D12, Metal and WebGPU. It enables a browser demo, and it costs less to maintain than raw Vulkan.
- **A 2D devour field per layer.** Destructible levels with no physics-debris cost.
- **Production discipline.** Steering gets the first two weeks. Readability rules are concrete. Validation gates have numeric exit criteria. The proposal names its own honest weakness and lists what it will not build.

**Weaknesses**
- **Originality.** It concedes the lowest originality score in its own set. "Dark interior of a dead machine" is a familiar sci-fi look, and it says less to non-gamers than a kitchen does.
- **A small growth fantasy.** 30,000 motes is modest next to "ten to a million". The trailer moment is weaker as a result.
- **No weak point.** With mass as the only health pool, the swarm plays as a blob. Formation matters less than it could.
- **Repetitive spaces.** Six colossi each have Hull, Arteries, Core and Heart. They risk feeling like one space reskinned six times.

---

## 3. Decision table

**Adopt** = take as written. **Adapt** = take, but changed. **Defer** = after launch. **Cut** = not built.

| Element | Source | Decision | Notes |
|---|---|---|---|
| Kitchen to city acts | 3 | **Adopt** | Five acts, each one step larger in scale. |
| Route choice between acts | 4 | **Adopt** | Each act has two variants, such as Kitchen/Pantry or Street/Market. |
| The Queen | 3 | **Adapt** | She becomes the hatchery as well (see §4.3). Losing her ends the run. |
| Mass as the only resource | 4 | **Adopt** | The visible count is the mass. Forms spend it. |
| Forms (combat shapes) | 4 | **Adopt** | Two Form slots. |
| Living structures | 3 | **Adapt** | Built as **traversal Forms**: Bridge, Tower and Raft on one context button. They use slot-assigned shapes first, not full PBD. |
| Species | 3 | **Adapt** | Species is the run's "character" and sets movement physics plus one signature trait. **4 at launch**, not 6. |
| Castes | 4 | **Adapt** | These are the castes inside a colony (Workers, Soldiers and so on). The Queen's laying ratio sets the mix. |
| Mutations with evolutions | 3 + 4 | **Adapt** | About 120 total in one shared pool, plus a few per species. Evolution combos follow the genre standard. |
| Adaptive city + Alarm | 3 + 4 | **Merge** | One **Response level**. It rises with mass and counters your dominant caste or damage type. Each counter is telegraphed. |
| Organ bosses with a Form counter | 4 | **Adopt** | Applied to Concept 3's boss list. |
| Colossus interior | 4 | **Adapt** | Becomes the **finale only**. The swarm breaches the Fumigator mech and eats it from inside. |
| 1M critters | 3 | **Adapt** | The count shown reaches 1M. About 32k agents carry the gameplay, and a cosmetic GPU layer renders the rest (see §5). |
| 3D voxel city + Jolt | 3 | **Cut → 2.5D** | A 2D material grid where each cell also has a height. Unsupported columns collapse into GPU debris. No Jolt. |
| Rust + wgpu | 4 | **Adopt** | Replaces C++20 on raw Vulkan. |
| Web demo (WebGPU) | 4 | **Adopt** | The Kitchen act. Agent counts there are small, so it suits browsers. |
| Local co-op + Confluence | 4 | **Adopt** | 2–4 players, plus Remote Play Together. |
| Hive Versus (online) | 3 | **Defer** | Goes through a post-launch gate (§10). The architecture keeps the door open. |
| Daily seeded city + leaderboards | 3 | **Adopt** | Ranked on time-to-a-million and percent consumed. |
| Pesticide / Hunger levels | 3 / 4 | **Adopt** | Uses the name **Pesticide levels**. |
| Sandbox + photo mode | 3 | **Adapt** | Photo mode ships at launch. The sandbox becomes the endless mode, set in a city. |
| Comic narrator | 3 | **Adopt** | An original voice and character, not an impression of a real presenter. It can be muted. |
| Macro → tilt-shift look | 3 | **Adopt** | Main art direction. |
| "Swarm is the light" | 4 | **Adapt** | Used for night acts (Town and City) and the Fireflies species. |
| Readability rules | 4 + 3 | **Merge** | A telegraph layer, a swarm that dims near threats, and a fixed color language. Adds the Queen beacon and audio threat cues. |
| Granular swarm audio | 3 + 4 | **Merge** | Music layers at 10k, 100k and 1M. Forms trigger chords. |
| Early access | 3 | **Cut** | Launch as 1.0 after a web demo and Next Fest. |

---

## 4. The combined game

### 4.1 Pitch
> **You don't fight the horde. You are the horde.** Hatch from ten critters on a kitchen floor. Shape the swarm into spears, shields and living bridges, guard your Queen, and eat your way to a million as the city falls.

**Taglines:** *Be the horde.* · *Ten to a million.* · *Guard the Queen. Eat the city.*

### 4.2 Pillars
1. **Growth you can see.** Every act is a new scale, so the camera pulls back and the world shrinks around you.
2. **The swarm's shape is the weapon.** Forms and structures are where player skill shows.
3. **One hunger, one weakness.** Mass is everything you have. The Queen is everything you can lose.
4. **Built for clips.** Every five seconds should be something worth sharing.

### 4.3 Core mechanics

**Controls.** Steer the swarm's focal point with the stick or WASD. The triggers **Gather** (tighten) and **Spread** (loosen). Two face buttons hold the two **combat Forms**. One button holds **Build**, which picks Bridge, Tower or Raft from context. That is all the input there is.

**Mass.** Every critter bites whatever it touches. Eating material and enemies adds mass. Losing critters loses mass. Forms cost mass and have cooldowns.

**The Queen (Concept 3's idea, reworked).**
- She follows the swarm's center with some lag, and enemies target her first.
- **New mass hatches at the Queen**, not at the front. Eating happens at the edges, but new critters are laid in the middle. Pushing the front far ahead means reinforcements arrive slowly. Keeping her close means exposing her.
- She has her own health and a short "last stand" window before the run ends. A sudden one-hit loss would feel unfair.
- A beacon, a silhouette outline and audio cues always show where she is. Queen mutations, such as faster laying or a carapace, form their own branch.

**Castes (Concept 4's idea, reframed as colony roles).** The Queen's **laying ratio** sets the mix of new hatches. Players adjust it at Bloom level-ups and at brood sites in the world.

| Caste | Role | Concept 4 equivalent |
|---|---|---|
| Workers | Eat fast and fill structures | Biters |
| Soldiers | Armored and slow; they block | Shells |
| Spitters | Acid damage over time | Burners |
| Fliers | Fast and fragile; they cross gaps | Gliders |
| Nurses | Turn mass into critters faster | Seeders |

**Species (Concept 3's idea, as the run's "character").** Each species sets movement physics and one signature trait, and gives the five castes its own look.

| Species (launch) | Physics | Signature |
|---|---|---|
| **Ants** | Strong grip; climbs walls | Structures are cheaper and carry more load |
| **Locusts** | Flight in short hops | Eats fastest, so Workers hit harder |
| **Slime cells** | Merge and flow; squeezes through gaps | Critters fuse into larger, tougher cells |
| **Fireflies** | Glow lights up the dark | Bloom detonates the glow, and night acts get brighter |

*After launch:* **Nanobots** (absorb machines, which suits the Fumigator finale) and one more species.

**Forms.**

| Form | Kind | Effect |
|---|---|---|
| Spear | Combat | Pierces armor in a line |
| Shield | Combat | A dome against spray, flame and gas |
| Ring | Combat | Encircles and crushes |
| Comet | Combat | A dash |
| Bloom | Combat | Bursts outward |
| Bridge | Build | Spans gaps and water |
| Tower | Build | Climbs walls and counters |
| Raft | Build | Floats on floods |

Combat Forms are distance fields in the swarm compute pass (Concept 4's method). Build Forms use the same shape targets. Critters snap into **lattice slots**, and how much load a structure carries depends on how many critters fill it. If prototyping shows that slot-snapping feels fake, position-based-dynamics constraints are the upgrade path.

**Mutations.** Devoured mass fills a **Bloom** meter. Each fill offers three cards. There are about 120 in total, with evolution combos such as *Acid Mandibles + Spore Bloom → Corrosive Bloom*. They modify castes, Forms, the Queen or a species trait.

**Response level (Concept 3's adaptive city merged with Concept 4's Alarm).**
- Pressure rises with mass.
- The city also counters your dominant damage type or caste. Lean on fire resistance and water cannons arrive; field a lot of fliers and nets appear.
- The narrator telegraphs every counter ("Remarkable. The city has noticed the acid.").

### 4.4 Run structure (about 25 minutes)

| Act | Count | Variants | Boss | Form counter |
|---|---|---|---|---|
| 1 Kitchen | 10 → 1k | Kitchen / Pantry | **The Cleaner**: spray cone and a wiping cloth | Shield blocks the spray; Spread slips under the wipe |
| 2 Garden | 1k → 10k | Garden / Basement | **Lawnmower** | Spread parts around the blades; Spear the fuel line |
| 3 Street | 10k → 50k | Street / Market | **Exterminator van**: fogger | Comet through the fog; Bridge the storm drain to flank |
| 4 Town (dusk) | 50k → 250k | Town / Docks | **Fire brigade**: hoses and floods | Raft rides the flood; Ring crushes the engine |
| 5 City (night) | 250k → 1M | Downtown / Harbor | **Fumigator mech**: gas sweeps, then a breach | Shield against the gas, Spear to breach, then **eat it from inside** across three short interior layers (Concept 4's colossus as the finale) |

Handcrafted chunks are assembled procedurally for each run. The run ends when the Queen dies or the Fumigator's core is devoured. The trailer moment, a counter ticking past 1,000,000 as a skyscraper falls, happens during Act 5.

### 4.5 Meta, modes and co-op
- **Evolution Codex.** It unlocks species, Forms, mutations, act variants and Strains (starting presets). It adds **breadth, not power**: there is no permanent stat grind (Concept 4's rule).
- **Pesticide levels.** Stacking difficulty modifiers.
- **Daily seeded city.** Leaderboards for time-to-a-million and percent consumed. This gives head-to-head competition with no netcode.
- **Endless sandbox city** with **photo mode**.
- **Local co-op, 2–4 players.** Each player has their own swarm, color and Queen. **Confluence** merges the swarms into giant Forms. Online play goes through Remote Play Together. The camera zooms to fit everyone, with a soft leash on stragglers.

### 4.6 Look, sound and tone
- **Visuals.** Macro photography at small scale: shallow depth of field, iridescent shells, warm sunlight. It eases into a wide tilt-shift look by Act 4. Acts 4–5 move to dusk and then night. There, Concept 4's "swarm as the light source" takes over, using a low-resolution light grid fed by swarm density.
- **Rendering by distance.** Near critters are animated meshes, which is cheap because Act 1 counts are small. Mid-range critters are impostors and far ones are points. Volumetric dust rises when buildings fall.
- **Tone.** Cartoon-macabre: critters burst into shell-flake confetti, with no gore. Each caste keeps the same color in every species, plus color-blind palettes.
- **Audio.** Granular synthesis tracks local density, rising from a chitter to a roar. Music adds layers at 10k, 100k and 1M. Forms trigger chords. Bosses attack in musical patterns.
- **Narrator.** A comic, original character who reacts to events ("Remarkable. The colony has found the fridge."). The VO script and budget are on the open-questions list (§11).

---

## 5. The scale model: an honest million

Concept 3 wants a million on screen. Concept 4 proves a budget of about 30k agents. We use both: **the gameplay agents are separate from what gets drawn.**

| Layer | What it is | Budget |
|---|---|---|
| **Gameplay agents** | Simulated critters that collide, bite, take damage and fill Forms | About 32k cap on every platform (Concept 4's tested budget) |
| **Critters per agent** | Grows by act: Kitchen 1, Garden 1, Street 4, Town 16, City 32 | Reaches 1M at the cap |
| **Cosmetic micro-swarm** | GPU-only particles that follow the agents' density and velocity fields; no effect on gameplay | Desktop up to 1M; Steam Deck about 250k; web is Kitchen only |

- Agents merge as critters-per-agent rises. This happens during the camera pull-back between acts, where the transition hides it.
- **Difficulty depends on mass, never on agent count**, so the challenge is the same on every hardware tier. Both proposals already make this point.
- The same split is the foundation for any later online mode. Versus would sync gameplay agents and density fields. Each client renders its own micro-swarm. This is Concept 3's netcode idea, reused.

---

## 6. Technical architecture

Concept 4's stack, with Concept 3's rendering and world ideas scaled to fit.

| Area | Choice | Why / source |
|---|---|---|
| Language | Rust | One codebase for desktop and web, and memory safety for a small team (4) |
| GPU | **wgpu** (Vulkan / D3D12 / Metal / WebGPU), WGSL shaders | Replaces raw Vulkan. Trade-off: no async compute queue, so budget compute serially (4 over 3) |
| Windowing / input | winit; gilrs + Steam Input | (4) |
| Swarm sim | GPU compute: spatial-hash flocking, with steering, Form/structure SDF targets and contact damage in one or two dispatches | (4) |
| Cosmetic layer | GPU particle advection over the agent density and velocity fields | New (§5) |
| World | **2.5D material grid**: each cell stores material, hardness and height. Agents erode it with atomic writes. A coarse CPU check over chunks finds unsupported columns, and those collapse into GPU debris | Concept 3's edible city at Concept 4's cost |
| Physics | Swarm rules plus simple collision for bosses and projectiles. Add **Rapier** only if collapses need real rigid chunks. No Jolt | (4); cuts Concept 3's C++ dependency |
| GPU → CPU | Async readbacks with one frame of latency; event buffers aggregated on the GPU | (4) |
| Gameplay | hecs ECS; content in data files | (4) |
| Rendering | Mesh → impostor → point LOD; depth-of-field and tilt-shift post; density light grid | (3) + (4) |
| Upscaling | FSR 1-style spatial upscaler ported to WGSL, for Steam Deck | (3), made portable |
| Audio | Kira plus a custom granular synthesizer | (4) |
| UI | Small immediate-mode UI; egui for debug tools | (4) |
| Platform | steamworks-rs: achievements, leaderboards, Steam Input, Remote Play Together | (4) |
| Assets | Blender → glTF, KTX2; a build script trims the web demo to under 25 MB | (4) |

**Performance targets (to validate):**
- 32k gameplay agents + 250k cosmetic particles at 60 fps on Steam Deck at 800p.
- 32k + 1M on a mid-range desktop GPU.
- Kitchen act at 60 fps in WebGPU browsers.

---

## 7. Launch content

| Content | Count |
|---|---|
| Acts | 5, each with 2 variants (10 maps from handcrafted chunks) |
| Bosses | 5, plus the Fumigator interior finale |
| Species | 4 |
| Castes | 5 |
| Forms | 8 (5 combat, 3 build) |
| Mutations | About 120, including evolutions |
| Modes | Run, Daily city, Pesticide levels, Endless sandbox + photo mode, local co-op |
| Platforms | Windows, Linux / Steam Deck (Verified target), macOS; web demo |

---

## 8. Commercial strategy

- **Price:** $9.99 premium, with **no early access** and **no microtransactions** (both proposals agree on price; Concept 4 on the rest).
- **Positioning:** "The survivors-like where you're the horde. Ten to a million."
- **Discovery:**
  - A **web demo of the Kitchen act** links straight from any clip.
  - A Steam page by month 4, then a Next Fest demo.
  - The **trailer hook**: the counter passes 1,000,000 as a skyscraper falls.
  - Narrator lines and daily seeds give streamers material.
- **After launch:** free updates (new species and act variants), a paid **city pack** for about $4.99, a soundtrack, and a Switch 2 port through a partner. Hive Versus free weekends only if Versus clears its gate.

---

## 9. What we are not taking

| Not taken | Why |
|---|---|
| 1M *simulated* agents | The cosmetic layer gives the same picture at roughly 3% of the simulation cost |
| 3D sparse voxel city + Jolt | 2.5D columns still deliver the falling skyscraper |
| C++20 on raw Vulkan, with ports later | Blocks the web demo and macOS; a heavier burden for a small team |
| PBD constraint structures (at first) | Slot-snapped structures come first; PBD only if they feel fake |
| Hive Versus at launch | Netcode cost plus empty-lobby risk; gated after launch |
| 6 species × 40 mutations each | 4 species and about 120 mutations in a shared pool |
| Six full colossi | One colossus interior, as the finale |
| Early access | 1.0 launch, validated through the demo and Next Fest |

---

## 10. Plan and gates

| Stage | Time and team | Scope | Exit criteria |
|---|---|---|---|
| **Minimum-fun prototype** | Weeks 1–8; 2–3 people. **Weeks 1–2 are steering only.** | Kitchen act; Queen with hatching at the Queen; 32k agents plus a cosmetic layer; 3 Forms (Spear, Shield, Bridge); edible material grid with one column collapse; Cleaner boss | Testers play 3+ runs without being asked. Most rate steering "good" or better. Strangers shown a 5-second clip say "you are the swarm". **Queen deaths feel fair** in post-run survey. Steam Deck performance target met |
| **Market validation** | Months 3–8; 4–5 people | Web demo of the Kitchen act; Steam page by month 4; trailer; Next Fest demo with Acts 1–2 | Demo-to-wishlist conversion of at least 20%; measurable web-to-Steam click-through; median demo session of at least 30 minutes |
| **Production** | Months 9–18, with buffer to month 20; 4–6 people | All of §7; narrator VO; Deck Verified; macOS | Launch between months 16 and 20 |
| **After launch** | Year one | Species updates; city pack; Switch 2 through a partner. **Hive Versus gate**: build it only if co-op and daily-seed engagement and concurrent players justify online matchmaking | Update engagement, pack attach rate, gate review at month 3 after launch |

The timeline is a little longer than Concept 4's 14–18 months. The added scope accounts for the difference: the Queen, structures, scale transitions with mesh LOD, the narrator and the finale interior. It is still shorter than Concept 3's 18–24 months to *early access*.

### Assumptions to validate before full production
1. Steering a swarm feels precise and satisfying (from 4).
2. Forms and Build add skill without breaking the genre's simple input (from 4).
3. **Hatching at the Queen creates real tension, and the Queen's death feels fair.**
4. **Scale transitions read clearly.** Players understand the camera pull-back and LOD changes, and the cosmetic million reads as "a million".
5. 32k agents plus the cosmetic layer hold 60 fps on Steam Deck.
6. Players can read threats amid the swarm (from 4).
7. The web demo turns browser visitors into wishlists (from 4).
8. $9.99 is the right price for this content.

---

## 11. Risks

| Type | Risk | Mitigation |
|---|---|---|
| Scope | Merging two proposals swells the scope | §9 lists every cut; gates in §10; no Versus at launch |
| Market | Crowded genre; GREX and browser swarm games got there first | Concept 3's scale curve, the Queen and structures raise originality above Concept 4 alone; ship the demo early |
| Design | Floaty swarm control | Two weeks of steering-only work; Gather and Spread; playtest gates |
| Design | The Queen's death feels cheap | Her own health pool, a last-stand window, the beacon and audio cues, Queen mutations |
| Design | Threats get buried in the swarm | A telegraph layer, dimming near danger, a fixed color language, the Queen beacon |
| Design | The cosmetic million feels fake, or players can't tell what is real | Cosmetic particles follow the gameplay agents tightly; damage and hits are drawn only on gameplay agents |
| Design | Depth runs out | Four build axes (species, castes, Forms, mutations); Response counters; route variants; daily seeds |
| Technical | wgpu has no async compute | Budget compute serially; keep the sim to one or two dispatches |
| Technical | GPU variety and browser support | Hardware tiers on the cosmetic layer only; WebGPU detection with a fallback page |
| Technical | 2.5D collapse looks cheap next to a voxel city | Debris particles, volumetric dust and a camera shake sell the fall; Rapier chunks are the upgrade path |
| Production | Narrator VO cost and iteration | Script in data files; temp VO until late production; must ship with a mute option and subtitles |
| Audience | Squeamish players | Cartoon-macabre treatment: shell-flake confetti, no gore |
| Commercial | The web demo cannibalizes sales | The demo holds only the Kitchen act |

---

## 12. Open questions for the team
1. **Narrator:** casting and budget, and whether the character has an on-screen identity.
2. **Species at launch:** 4 (recommended) or 6. Two more add about 3 months of content and balance work.
3. **Finale interior:** in 1.0 (recommended, because it is the bridge to Concept 4's best set piece), or cut to a straight boss fight if production slips.
4. **Hive Versus gate:** what concurrent-player threshold justifies building online versus.
5. **Name:** both concepts use *Myriad*. Check trademark and Steam search clutter before the Steam page goes live in month 4.
