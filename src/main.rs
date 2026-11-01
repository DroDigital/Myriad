use myriad::cli::{self, Command};

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn,myriad=info"))
        .init();
    match cli::parse(std::env::args().skip(1))? {
        Command::Help => println!("{}", cli::USAGE),
        Command::Headless(opts) => println!("{}", myriad::headless::run(&opts)?),
        Command::Windowed(opts) => myriad::app::run(opts)?,
    }
    Ok(())
}
