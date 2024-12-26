use sealy::{
	BFVEncoder, BFVEncryptionParametersBuilder, Evaluator, CoefficientModulusFactory, Context,
	Decryptor, DegreeType, Encryptor, Evaluator, KeyGenerator, PlainModulusFactory, SecurityLevel,
};

fn main() -> anyhow::Result<()> {
	let params = BFVEncryptionParametersBuilder::new()
		.set_poly_modulus_degree(DegreeType::D8192)
		.set_coefficient_modulus(
			CoefficientModulusFactory::build(DegreeType::D8192, &[50, 30, 30, 50, 50]).unwrap(),
		)
		.set_plain_modulus(PlainModulusFactory::batching(DegreeType::D8192, 32)?)
		.build()?;

	let ctx = Context::new(&params, false, SecurityLevel::TC128)?;
	let gen = KeyGenerator::new(&ctx)?;

	let encoder = BFVEncoder::new(&ctx)?;

	let public_key = gen.create_public_key();
	let secret_key = gen.secret_key();

	let encryptor = Encryptor::with_public_key(&ctx, &public_key)?;
	let decryptor = Decryptor::new(&ctx, &secret_key)?;
	let evaluator = Evaluator::new(&ctx)?;

	let plaintext: Vec<i64> = vec![1, 2, 3];
	let factor = vec![2, 2, 2];

	let encoded_plaintext = encoder.encode_i64(&plaintext)?;
	let encoded_factor = encoder.encode_i64(&factor)?;

	let ciphertext = encryptor.encrypt(&encoded_plaintext)?;
	let ciphertext_result = evaluator.multiply_plain(&ciphertext, &encoded_factor)?;

	let decrypted = decryptor.decrypt(&ciphertext_result)?;
	let decoded = encoder.decode_i64(&decrypted);

	println!("{:?}", &decoded.into_iter().take(3).collect::<Vec<_>>()); // [2, 4, 6]

	Ok(())
}
