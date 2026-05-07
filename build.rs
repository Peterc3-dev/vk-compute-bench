use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let shader_dir = Path::new("shaders");

    let shaders = ["saxpy.comp", "reduce.comp", "matmul.comp", "copy.comp"];

    for shader in &shaders {
        let input = shader_dir.join(shader);
        let stem = shader.replace(".comp", "");
        let output = Path::new(&out_dir).join(format!("{}.spv", stem));

        println!("cargo:rerun-if-changed={}", input.display());

        let status = Command::new("glslangValidator")
            .args([
                "-V",
                "--target-env",
                "vulkan1.1",
                "-o",
                output.to_str().unwrap(),
                input.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to run glslangValidator. Is it installed?");

        if !status.success() {
            panic!("Shader compilation failed for {}", shader);
        }
    }
}
