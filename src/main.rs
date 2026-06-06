use clap::{Parser, Subcommand};
use bitnet::{Engine, EngineConfig, GenerateConfig};
use bitnet::quantization::QuantMode;

#[derive(Debug, Parser)]
#[command(
    name = "bitnet-cli",
    about = "1-bit / 1.58-bit quantisation-native LLM inference engine"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run inference on a prompt
    Generate {
        #[arg(short, long)]
        model: String,
        #[arg(short, long)]
        prompt: String,
        #[arg(long, default_value = "256")]
        max_tokens: usize,
        #[arg(long, default_value = "0.7")]
        temperature: f32,
        #[arg(long, default_value = "ternary")]
        quant: String,
    },
    /// Print memory estimate for a model configuration
    MemEstimate {
        #[arg(long, default_value = "32")]
        layers: usize,
        #[arg(long, default_value = "4096")]
        d_model: usize,
        #[arg(long, default_value = "11008")]
        d_ff: usize,
        #[arg(long, default_value = "32000")]
        vocab: usize,
        #[arg(long, default_value = "ternary")]
        quant: String,
    },
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.cmd {
        Command::Generate {
            model,
            prompt,
            max_tokens,
            temperature,
            quant,
        } => {
            let mode = parse_quant(&quant)?;
            let cfg = EngineConfig {
                model_path: model.into(),
                quant_mode: mode,
                ..Default::default()
            };
            let engine = Engine::from_config(cfg)?;
            let gen_cfg = GenerateConfig {
                max_new_tokens: max_tokens,
                temperature,
                ..Default::default()
            };
            let output = engine.generate(&prompt, gen_cfg)?;
            println!("{}", output);
        }
        Command::MemEstimate {
            layers,
            d_model,
            d_ff,
            vocab,
            quant,
        } => {
            let mode = parse_quant(&quant)?;
            let bytes = bitnet::utils::memory::estimate_model_memory(layers, d_model, d_ff, vocab, mode);
            println!(
                "Estimated memory: {:.2} GB ({} bytes)",
                bytes as f64 / 1e9,
                bytes
            );
        }
    }

    Ok(())
}

fn parse_quant(s: &str) -> anyhow::Result<QuantMode> {
    match s {
        "binary" | "1bit" | "1-bit" => Ok(QuantMode::Binary),
        "ternary" | "1.58bit" | "158bit" => Ok(QuantMode::Ternary),
        other => anyhow::bail!("Unknown quant mode: {}. Use binary or ternary", other),
    }
}
