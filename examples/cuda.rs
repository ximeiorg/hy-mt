//! Prints the device that would be selected, and exercises an explicit one.
//!
//! ```bash
//! cargo run --release --features cuda --example cuda
//! ```

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = hy_mt::default_device()?;
    println!("auto   : {device:?}");
    println!("cpu    : {:?}", hy_mt::device_from_str("cpu")?);
    #[cfg(feature = "cuda")]
    println!("cuda:0 : {:?}", hy_mt::device_from_str("cuda:0")?);
    #[cfg(feature = "metal")]
    println!("metal:0: {:?}", hy_mt::device_from_str("metal:0")?);
    Ok(())
}
