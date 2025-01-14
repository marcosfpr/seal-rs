use criterion::{black_box, criterion_group, criterion_main, Criterion};

use rand::Rng;
use sealy::{
	BFVEncoder, BFVEncryptionParametersBuilder, Ciphertext, CoefficientModulusFactory, Context,
	DegreeType, EncryptionParameters, Encryptor, Error, Evaluator, Evaluator, KeyGenerator,
	PlainModulusFactory, SecurityLevel,
};

fn generate_clients_gradients(
	num_clients: usize,
	tensor_dim: usize,
) -> Vec<Vec<f64>> {
	let mut clients = Vec::with_capacity(num_clients);
	for _ in 0..num_clients {
		let mut tensor = Vec::with_capacity(tensor_dim);
		for _ in 0..tensor_dim {
			tensor.push(rand::thread_rng().gen_range(0.0..1.0));
		}
		clients.push(tensor);
	}
	clients
}

fn create_bfv_context(
	degree: DegreeType,
	bit_size: u32,
) -> Result<Context, Error> {
	let security_level = SecurityLevel::TC128;
	let expand_mod_chain = false;
	let modulus_chain = CoefficientModulusFactory::bfv(degree, security_level)?;
	let encryption_parameters: EncryptionParameters = BFVEncryptionParametersBuilder::new()
		.set_poly_modulus_degree(degree)
		.set_plain_modulus(PlainModulusFactory::batching(degree, bit_size)?)
		.set_coefficient_modulus(modulus_chain.clone())
		.build()?;

	let ctx = Context::new(&encryption_parameters, expand_mod_chain, security_level)?;

	Ok(ctx)
}

fn aggregate(
	ctx: &Context,
	encoder: &BFVEncoder,
	base: f64,
	ciphertexts: &[Ciphertext],
	dimension: usize,
) -> Result<Ciphertext, Error> {
	let evaluator = Evaluator::new(ctx)?;
	let cipher = evaluator.add_many(ciphertexts)?;

	let fraction = 1.0 / ciphertexts.len() as f64;
	let fraction = vec![fraction; dimension];
	let fraction = encoder.encode_f64(&fraction, base)?;

	evaluator.multiply_plain(&cipher, &fraction)
}

fn criterion_benchmark(c: &mut Criterion) {
	let dimension = 16_384;
	let num_clients = 10;
	let base = 1_000_000_000f64;

	println!("dimension: {}", dimension);
	println!("num_clients: {}", num_clients);

	print!("Generating clients gradients...");
	let clients = generate_clients_gradients(num_clients, dimension);
	println!("done");

	print!("Creating BFV context...");
	let ctx = create_bfv_context(DegreeType::D32768, 60).expect("Failed to create BFV context");
	println!("done");

	let key_gen = KeyGenerator::new(&ctx).expect("Failed to create key generator");
	let encoder = BFVEncoder::new(&ctx).expect("Failed to create encoder");

	let public_key = key_gen.create_public_key();
	let private_key = key_gen.secret_key();

	println!("Encoding clients gradients...");
	let mut plaintexts = Vec::with_capacity(num_clients);
	for client in clients.iter() {
		let encoded = encoder
			.encode_f64(client, base)
			.expect("Failed to encode client gradients");
		plaintexts.push(encoded);
	}

	println!("Encrypting clients gradients...");
	let mut ciphertexts = Vec::with_capacity(num_clients);
	let encryptor = Encryptor::with_public_and_secret_key(&ctx, &public_key, &private_key)
		.expect("Failed to create encryptor");
	for plaintext in plaintexts.iter() {
		let ciphertext = encryptor
			.encrypt(plaintext)
			.expect("Failed to encrypt client gradients");
		ciphertexts.push(ciphertext);
	}

	println!("Benchmarking BFV aggregate 16k...");
	c.bench_function("aggregate 16k BFV", |b| {
		b.iter(|| {
			aggregate(
				black_box(&ctx),
				black_box(&encoder),
				black_box(base),
				black_box(&ciphertexts),
				black_box(dimension),
			)
		})
	});
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
