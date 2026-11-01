//! HUD layout: turns run state into positioned glyphs. Pure CPU code.

use crate::queen::{MAX_HEALTH, QueenState};
use crate::render::GlyphInstance;
use crate::render::font;
use crate::run::{Run, RunState};

/// Palette indices understood by the text shader.
pub mod color {
    pub const WHITE: u32 = 0;
    pub const AMBER: u32 = 1;
    pub const RED: u32 = 2;
    pub const DIM: u32 = 3;
    pub const GREEN: u32 = 4;
}

/// Font pixels -> screen pixels at 720p; scaled with the viewport height.
fn base_scale(viewport_h: f32) -> f32 {
    (viewport_h / 720.0 * 3.0).max(2.0).round()
}

#[derive(Default)]
pub struct TextBatch {
    pub glyphs: Vec<GlyphInstance>,
}

impl TextBatch {
    /// Width in pixels of `s` at `scale`.
    pub fn width(s: &str, scale: f32) -> f32 {
        s.chars().count() as f32 * (font::COLS as f32 + 1.0) * scale
    }

    pub fn text(&mut self, x: f32, y: f32, s: &str, color: u32, scale: f32) {
        let advance = (font::COLS as f32 + 1.0) * scale;
        for (i, c) in s.chars().enumerate() {
            if c == ' ' {
                continue;
            }
            self.glyphs.push(GlyphInstance {
                pos: [x + i as f32 * advance, y],
                glyph: font::glyph(c),
                style: color | ((scale as u32).min(255) << 8),
            });
        }
    }

    pub fn centered(&mut self, cx: f32, y: f32, s: &str, color: u32, scale: f32) {
        self.text(cx - Self::width(s, scale) * 0.5, y, s, color, scale);
    }
}

/// Lays out the HUD for `run` on a `w` x `h` viewport.
pub fn build(run: &Run, w: f32, h: f32) -> TextBatch {
    let mut b = TextBatch::default();
    let s = base_scale(h);
    let big = s * 2.0;
    let line = 9.0 * s;
    let margin = 4.0 * s;

    // Mass, the one number that matters.
    let mass = run.colony.mass();
    b.text(margin, margin, &format!("MASS {mass}"), color::WHITE, big);
    b.text(
        margin,
        margin + 2.2 * line,
        &format!("CRITTERS {}   LOST {}", run.colony.alive, run.colony.lost),
        color::DIM,
        s,
    );

    // The Queen's health as a bar of blocks.
    let q = &run.queen;
    let filled = ((q.health / MAX_HEALTH) * 10.0).ceil().clamp(0.0, 10.0) as usize;
    let bar: String = std::iter::repeat_n('\u{2588}', filled).collect();
    let bar_color = match q.state {
        QueenState::Healthy if q.health > 40.0 => color::GREEN,
        QueenState::Healthy => color::AMBER,
        _ => color::RED,
    };
    let y = margin + 3.4 * line;
    b.text(margin, y, "QUEEN", color::AMBER, s);
    b.text(
        margin + TextBatch::width("QUEEN ", s),
        y,
        &bar,
        bar_color,
        s,
    );

    // Form slots, bottom left: key, name, cost or cooldown.
    let mass = run.colony.mass();
    let keys = ["Z", "X"];
    for (slot, kind) in run.forms.slots.iter().enumerate() {
        let def = kind.def();
        let (state, color) = match run.forms.active {
            Some(a) if a.slot == slot => ("ACTIVE".to_string(), color::GREEN),
            _ if run.forms.cooldown[slot] > 0.0 => (
                format!("{:.0}S", run.forms.cooldown[slot].ceil()),
                color::DIM,
            ),
            _ if kind.cost(mass) >= mass => ("TOO FEW".to_string(), color::DIM),
            _ => (format!("-{}", kind.cost(mass)), color::AMBER),
        };
        let y = h - margin - (3.0 - slot as f32) * line * 1.2;
        b.text(
            margin,
            y,
            &format!("{} {}", keys[slot], def.name),
            color::WHITE,
            s,
        );
        b.text(
            margin + TextBatch::width("Z SHIELD  ", s),
            y,
            &state,
            color,
            s,
        );
    }
    let build_color = if run.building {
        color::GREEN
    } else {
        color::DIM
    };
    b.text(margin, h - margin - line * 1.2, "B BUILD", build_color, s);

    // Level progress, top right.
    let pct = format!("KITCHEN {:.0}% EATEN", run.eaten_fraction() * 100.0);
    b.text(
        w - margin - TextBatch::width(&pct, s),
        margin,
        &pct,
        color::DIM,
        s,
    );

    // Warnings and outcomes, centred.
    let blink = (run.game.time * 3.0).fract() < 0.6;
    if let Some(mop) = &run.mop
        && mop.warning() > 0.0
        && blink
    {
        b.centered(w * 0.5, margin, "THE MOP IS COMING", color::AMBER, s);
    }
    if matches!(q.state, QueenState::LastStand { .. }) && blink {
        b.centered(w * 0.5, h * 0.3, "THE QUEEN IS IN DANGER!", color::RED, big);
    }
    if run.state == RunState::Lost {
        b.centered(w * 0.5, h * 0.38, "THE QUEEN HAS FALLEN", color::RED, big);
        b.centered(
            w * 0.5,
            h * 0.38 + 2.5 * line,
            &format!(
                "PEAK MASS {}  -  PRESS R TO TRY AGAIN",
                run.colony.peak_mass
            ),
            color::WHITE,
            s,
        );
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::level::Level;
    use crate::run::RunConfig;

    fn text_of(b: &TextBatch) -> String {
        b.glyphs
            .iter()
            .map(|g| char::from_u32(g.glyph).unwrap_or('?'))
            .collect()
    }

    #[test]
    fn shows_mass_and_queen() {
        let run = Run::new(&Level::kitchen(1), &RunConfig::default(), 16.0 / 9.0);
        let b = build(&run, 1280.0, 720.0);
        let t = text_of(&b);
        assert!(t.contains("MASS10"), "{t}");
        assert!(t.contains("QUEEN"));
        // A full health bar is 10 blocks.
        let blocks = b
            .glyphs
            .iter()
            .filter(|g| g.glyph == font::BLOCK as u32)
            .count();
        assert_eq!(blocks, 10);
        // Everything is on screen.
        assert!(
            b.glyphs
                .iter()
                .all(|g| g.pos[0] >= 0.0 && g.pos[0] < 1280.0)
        );
    }

    #[test]
    fn shows_form_slots_and_build() {
        let run = Run::new(&Level::kitchen(1), &RunConfig::default(), 16.0 / 9.0);
        let t = text_of(&build(&run, 1280.0, 720.0));
        assert!(t.contains("ZSPEAR"), "{t}");
        assert!(t.contains("XSHIELD"));
        assert!(t.contains("-8"), "a Shield costs 8 at mass 10: {t}");
        assert!(t.contains("BBUILD"));
        // A swarm of 5 cannot pay for either.
        let mut tiny = Run::new(&Level::kitchen(1), &RunConfig::default(), 16.0 / 9.0);
        tiny.colony.alive = 5;
        let t = text_of(&build(&tiny, 1280.0, 720.0));
        assert_eq!(t.matches("TOOFEW").count(), 2, "{t}");
    }

    #[test]
    fn game_over_message() {
        let mut run = Run::new(&Level::kitchen(1), &RunConfig::default(), 16.0 / 9.0);
        run.state = RunState::Lost;
        let t = text_of(&build(&run, 1280.0, 720.0));
        assert!(t.contains("THEQUEENHASFALLEN"), "{t}");
        assert!(t.contains("PRESSRTOTRYAGAIN"));
    }
}
