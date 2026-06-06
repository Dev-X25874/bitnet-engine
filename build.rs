use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/");

    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        compile_cuda_kernels();
    }

    if env::var("CARGO_FEATURE_AVX512").is_ok() {
        println!("cargo:rustc-cfg=feature=\"avx512\"");
    }
    if env::var("CARGO_FEATURE_NEON").is_ok() {
        println!("cargo:rustc-cfg=feature=\"neon\"");
    }
}

fn compile_cuda_kernels() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cuda_dir = PathBuf::from("cuda/kernels");

    let cu_files: Vec<PathBuf> = fs::read_dir(&cuda_dir)
        .expect("cuda/kernels directory not found")
        .filter_map(|e| {
            let path = e.ok()?.path();
            if path.extension()?.to_str()? == "cu" {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    let mut obj_files: Vec<PathBuf> = Vec::new();

    for cu in &cu_files {
        let stem = cu.file_stem().unwrap().to_str().unwrap();
        let obj = out_dir.join(format!("{}.o", stem));

        let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_string());

        let status = Command::new("nvcc")
            .args(["-O3", &format!("-arch={}", arch)])
            .args(["--use_fast_math", "-Xptxas=-v"])
            .arg("-Icuda/include")
            .args(["-c", cu.to_str().unwrap()])
            .arg("-o")
            .arg(&obj)
            .status()
            .expect("nvcc not found — install CUDA toolkit and ensure nvcc is on PATH");

        assert!(status.success(), "nvcc failed for {:?}", cu);
        obj_files.push(obj);
    }

    let lib = out_dir.join("libbitnet_cuda.a");
    let mut ar = Command::new("ar");
    ar.arg("crs").arg(&lib);
    for obj in &obj_files {
        ar.arg(obj);
    }
    ar.status().expect("ar command failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=bitnet_cuda");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
