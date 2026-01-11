use candle_core::{Device, Result};
use hy_mt::TextGeneration;

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;

    println!("设备: {:?}", &device);

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tencent/HY-MT1.5-1.8B");

    let mut model = TextGeneration::from_pretrained(path, &device)?;


    let prompt = "Translate the following segment into Chinese without additional explanation.\n\nIt’s on the house.";

    let result = model.generate(prompt, 100)?;
    println!("结果:{result}");

    Ok(())
}
