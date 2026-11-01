//! Command-line parsing (kept dependency-free).

use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use crate::app::AppOptions;
use crate::headless::HeadlessOptions;

pub const USAGE: &str = "\
Usage: myriad [OPTIONS]

Options:
  --agents N         critters to start with (default: 2000 windowed, 20000 headless)
  --capacity N       maximum critters (default 32768)
  --seed N           random seed (default 1)
  --headless         run without a window: scripted steering, timing and stats
  --frames N         headless: simulation steps to run (default 600)
  --width N          headless: screenshot width (default 1280)
  --height N         headless: screenshot height (default 720)
  --screenshot PATH  headless: save the final frame as a PNG
  -h, --help         show this help

Controls (windowed):
  WASD / arrows  steer the focal point     left mouse  steer toward cursor
  Space          Gather                    Shift       Spread
  H              hatch 1000 critters       Esc         quit";

#[derive(Debug)]
pub enum Command {
    Windowed(AppOptions),
    Headless(HeadlessOptions),
    Help,
}

pub fn parse<I: IntoIterator<Item = String>>(args: I) -> Result<Command> {
    let mut headless = false;
    let mut agents = None;
    let mut capacity = 32_768;
    let mut seed = 1;
    let mut h = HeadlessOptions::default();
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().with_context(|| format!("{name} needs a value"));
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--headless" => headless = true,
            "--agents" => agents = Some(number(&value("--agents")?)?),
            "--capacity" => capacity = number(&value("--capacity")?)?,
            "--seed" => seed = number(&value("--seed")?)?,
            "--frames" => h.frames = number(&value("--frames")?)?,
            "--width" => h.width = number(&value("--width")?)?,
            "--height" => h.height = number(&value("--height")?)?,
            "--screenshot" => h.screenshot = Some(PathBuf::from(value("--screenshot")?)),
            other => bail!("unknown argument `{other}`\n\n{USAGE}"),
        }
    }
    if headless {
        h.agents = agents.unwrap_or(h.agents);
        h.capacity = capacity;
        h.seed = seed;
        if h.width == 0 || h.height == 0 {
            bail!("--width and --height must be positive");
        }
        Ok(Command::Headless(h))
    } else {
        Ok(Command::Windowed(AppOptions {
            agents: agents.unwrap_or(2_000),
            capacity,
            seed,
        }))
    }
}

fn number<T: std::str::FromStr>(s: &str) -> Result<T> {
    s.replace('_', "")
        .parse()
        .map_err(|_| anyhow::anyhow!("`{s}` is not a valid number"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn default_is_windowed() {
        match parse(args("")).unwrap() {
            Command::Windowed(o) => assert_eq!(o.agents, 2_000),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn headless_options() {
        match parse(args(
            "--headless --agents 10_000 --frames 5 --screenshot out/a.png",
        ))
        .unwrap()
        {
            Command::Headless(h) => {
                assert_eq!(h.agents, 10_000);
                assert_eq!(h.frames, 5);
                assert_eq!(h.screenshot.unwrap(), PathBuf::from("out/a.png"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn rejects_unknown_and_missing_values() {
        assert!(parse(args("--bogus")).is_err());
        assert!(parse(args("--agents")).is_err());
        assert!(parse(args("--agents lots")).is_err());
    }
}
