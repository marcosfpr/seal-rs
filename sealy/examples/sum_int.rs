use sealy::{
	BFVEncoder, BFVEncryptionParametersBuilder, Evaluator, CoefficientModulusFactory, Context,
	Decryptor, DegreeType, EncryptionParameters, Encryptor, Evaluator, KeyGenerator,
	PlainModulusFactory, SecurityLevel,
};

fn main() -> anyhow::Result<()> {
	// generate keypair to encrypt and decrypt data.
	let degree = DegreeType::D8192;
	let bit_size = 60;
	let security_level = SecurityLevel::TC128;

	let expand_mod_chain = false;
	let encryption_parameters: EncryptionParameters = BFVEncryptionParametersBuilder::new()
		.set_poly_modulus_degree(DegreeType::D8192)
		.set_coefficient_modulus(CoefficientModulusFactory::bfv(degree, security_level)?)
		.set_plain_modulus(PlainModulusFactory::batching(degree, bit_size)?)
		.build()?;

	let ctx = Context::new(&encryption_parameters, expand_mod_chain, security_level)?;

	let key_gen = KeyGenerator::new(&ctx)?;
	let encoder = BFVEncoder::new(&ctx)?;

	let public_key = key_gen.create_public_key();
	let private_key = key_gen.secret_key();

	let encryptor = Encryptor::with_public_and_secret_key(&ctx, &public_key, &private_key)?;
	let decryptor = Decryptor::new(&ctx, &private_key)?;

	let evaluator = Evaluator::new(&ctx)?;

	let x = 5000001231231313;
	let y = 1000123123132131;

	let x_encoded = encoder.encode_i64(&[x])?;
	let y_encoded = encoder.encode_i64(&[y])?;

	let x_enc = encryptor.encrypt(&x_encoded)?;
	let y_enc = encryptor.encrypt(&y_encoded)?;

	let sum = evaluator.add(&x_enc, &y_enc)?;
	let sum_dec = decryptor.decrypt(&sum)?;
	let sum_dec = encoder.decode_i64(&sum_dec)?;

	let truth = x + y;
	let result = sum_dec.first().unwrap();
	assert_eq!(result, &truth);

	println!("truth: {:?}", truth);
	println!("sum: {:?}", result);

	Ok(())
}
