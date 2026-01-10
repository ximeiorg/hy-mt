use candle_core::Device;

fn main () {
    let device = Device::cuda_if_available(0).unwrap();
    println!("设备: {:?}", &device);
}