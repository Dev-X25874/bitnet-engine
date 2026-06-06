use criterion::{
    black_box, criterion_group, criterion_main,
    BenchmarkId, Criterion, Throughput,
};
use bitnet::quantization::{BitPacking, PackedMatrix, QuantMode};
use bitnet::quantization::scale::quantise_activation;
use bitnet::ops::cpu_gemv::bitgemv_cpu;

// ---------------------------------------------------------------------------
// Binary GEMV
// ---------------------------------------------------------------------------

fn bench_binary_gemv(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_gemv");

    for &(n, k) in &[
        (4096,  4096),
        (11008, 4096),
        (4096,  11008),
    ] {
        let w_data: Vec<f32> = (0..n * k)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let x_data: Vec<f32> = (0..k)
            .map(|i| (i as f32).sin())
            .collect();

        let packed              = PackedMatrix::pack_f32(&w_data, n, k, QuantMode::Binary).unwrap();
        let (x_packed, x_scale) = quantise_activation(&x_data);
        let mut y               = vec![0.0f32; n];

        group.throughput(Throughput::Elements((2 * n * k) as u64));
        group.bench_with_input(
            BenchmarkId::new("cpu", format!("{}x{}", n, k)),
            &(n, k),
            |b, _| {
                b.iter(|| {
                    y.iter_mut().for_each(|v| *v = 0.0);
                    bitgemv_cpu(
                        black_box(&packed),
                        black_box(&x_packed),
                        black_box(x_scale),
                        black_box(&mut y),
                    );
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Ternary GEMV
// ---------------------------------------------------------------------------

fn bench_ternary_gemv(c: &mut Criterion) {
    let mut group = c.benchmark_group("ternary_gemv");

    for &(n, k) in &[
        (4096,  4096),
        (11008, 4096),
    ] {
        let w_data: Vec<f32> = (0..n * k)
            .map(|i| match i % 3 { 0 => 1.0, 1 => -1.0, _ => 0.0 })
            .collect();
        let x_data: Vec<f32> = (0..k)
            .map(|i| (i as f32 * 0.01).tanh())
            .collect();

        let packed              = PackedMatrix::pack_f32(&w_data, n, k, QuantMode::Ternary).unwrap();
        let (x_packed, x_scale) = quantise_activation(&x_data);
        let mut y               = vec![0.0f32; n];

        group.throughput(Throughput::Elements((2 * n * k) as u64));
        group.bench_with_input(
            BenchmarkId::new("cpu", format!("{}x{}", n, k)),
            &(n, k),
            |b, _| {
                b.iter(|| {
                    y.iter_mut().for_each(|v| *v = 0.0);
                    bitgemv_cpu(
                        black_box(&packed),
                        black_box(&x_packed),
                        black_box(x_scale),
                        black_box(&mut y),
                    );
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Weight packing
// ---------------------------------------------------------------------------

fn bench_pack(c: &mut Criterion) {
    let mut group = c.benchmark_group("weight_packing");

    let n = 4096; let k = 4096;
    let w_data: Vec<f32> = (0..n * k)
        .map(|i| (i as f32).sin())
        .collect();

    group.throughput(Throughput::Bytes((n * k * 4) as u64));

    group.bench_function("binary_pack_4096x4096", |b| {
        b.iter(|| {
            PackedMatrix::pack_f32(
                black_box(&w_data), n, k, QuantMode::Binary
            ).unwrap()
        })
    });

    group.bench_function("ternary_pack_4096x4096", |b| {
        b.iter(|| {
            PackedMatrix::pack_f32(
                black_box(&w_data), n, k, QuantMode::Ternary
            ).unwrap()
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Activation quantisation
// ---------------------------------------------------------------------------

fn bench_quantise_activation(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantise_activation");

    for &k in &[1024usize, 4096, 11008] {
        let x_data: Vec<f32> = (0..k).map(|i| (i as f32).cos()).collect();
        group.throughput(Throughput::Bytes((k * 4) as u64));
        group.bench_with_input(
            BenchmarkId::new("cpu", k),
            &k,
            |b, _| {
                b.iter(|| quantise_activation(black_box(&x_data)))
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_binary_gemv,
    bench_ternary_gemv,
    bench_pack,
    bench_quantise_activation,
);
criterion_main!(benches);
