use thorn_seal::{
	BFVEncoder, BFVEvaluator, BfvEncryptionParametersBuilder, CoefficientModulus, Context,
	Decryptor, EncryptionParameters, Encryptor, Evaluator, KeyGenerator, PlainModulus,
	SecurityLevel,
};

fn main() -> anyhow::Result<()> {
	// generate keypair to encrypt and decrypt data.
	let degree = 8192;
	let lane_bits = 17;
	let security_level = SecurityLevel::TC128;

	let expand_mod_chain = false;
	let encryption_parameters: EncryptionParameters = BfvEncryptionParametersBuilder::new()
		.set_poly_modulus_degree(degree)
		.set_coefficient_modulus(CoefficientModulus::bfv_default(degree, security_level)?)
		.set_plain_modulus(PlainModulus::batching(degree, lane_bits)?)
		.build()?;

	let ctx = Context::new(&encryption_parameters, expand_mod_chain, security_level)?;

	let key_gen = KeyGenerator::new(&ctx)?;
	let encoder = BFVEncoder::new(&ctx)?;

	let public_key = key_gen.create_public_key();
	let private_key = key_gen.secret_key();

	let encryptor = Encryptor::with_public_and_secret_key(&ctx, &public_key, &private_key)?;
	let decryptor = Decryptor::new(&ctx, &private_key)?;

	let evaluator = BFVEvaluator::new(&ctx)?;

	let x = 5;
	let y = 10;

	let x_enc = encryptor.encrypt(&encoder.encode_signed(&[x])?)?;
	let y_enc = encryptor.encrypt(&encoder.encode_signed(&[y])?)?;

	println!("Summing x + y...");
	println!("x: {:#?}", x_enc);
	println!("y: {:#?}", y_enc);

	let sum = evaluator.add(&x_enc, &y_enc)?;
	let sum_dec = decryptor.decrypt(&sum)?;

	let sum_dec = encoder.decode_signed(&sum_dec)?;

	println!("Sum: {:?}", sum_dec.first());

	Ok(())
}
