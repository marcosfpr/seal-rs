use sealy::{
	CKKSEncoder, CKKSEncryptionParametersBuilder, Evaluator, CoefficientModulusFactory,
	Context, Decryptor, DegreeType, EncryptionParameters, Encryptor, Error, Evaluator,
	KeyGenerator, SecurityLevel,
};

fn main() -> Result<(), Error> {
	// generate keypair to encrypt and decrypt data.
	let degree = DegreeType::D8192;
	let security_level = SecurityLevel::TC128;
	let bit_sizes = [60, 40, 40, 60];

	let expand_mod_chain = false;
	let modulus_chain = CoefficientModulusFactory::build(degree, bit_sizes.as_slice())?;
	let encryption_parameters: EncryptionParameters = CKKSEncryptionParametersBuilder::new()
		.set_poly_modulus_degree(degree)
		.set_coefficient_modulus(modulus_chain.clone())
		.build()?;

	let ctx = Context::new(&encryption_parameters, expand_mod_chain, security_level)?;

	let key_gen = KeyGenerator::new(&ctx)?;
	let encoder = CKKSEncoder::new(&ctx, 2.0f64.powi(40))?;

	let public_key = key_gen.create_public_key();
	let private_key = key_gen.secret_key();

	let encryptor = Encryptor::with_public_and_secret_key(&ctx, &public_key, &private_key)?;
	let decryptor = Decryptor::new(&ctx, &private_key)?;

	let evaluator = Evaluator::new(&ctx)?;

	let x = 5.2;
	let y = 3.3;

	let x_encoded = encoder.encode_f64(&[x])?;
	let y_encoded = encoder.encode_f64(&[y])?;

	let x_enc = encryptor.encrypt(&x_encoded)?;
	let y_enc = encryptor.encrypt(&y_encoded)?;

	let sum = evaluator.add(&x_enc, &y_enc)?;
	let sum_dec = decryptor.decrypt(&sum)?;
	let sum_dec = encoder.decode_f64(&sum_dec)?;

	let truth = x + y;

	// Compare with a tolerance of 1e-6
	let result = sum_dec.first().unwrap();
	assert!((result - truth).abs() < 1e-6);

	println!("truth: {:?}", truth);
	println!("sum: {:?}", result);

	Ok(())
}
