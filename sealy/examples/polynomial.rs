use sealy::{
	CKKSEncoder, Ciphertext, CoefficientModulusFactory, CoefficientModulusType, Context, Decryptor, EncryptionParameters, Encryptor, Evaluator, EvaluatorOps, KeyGenerator, Plaintext, RelinearizationKey, SchemeType, SecretKey, SecurityLevel
};
use std::f64::consts::PI;

fn main() -> anyhow::Result<()> {
	println!("Example: Evaluating polynomial PI*x^3 + 0.4x + 1 using CKKS");

	let poly_modulus_degree = 8192;
	let modulus_chain = CoefficientModulusFactory::create(poly_modulus_degree, &[60, 40, 40, 60])?;

	let encryption_parameters: EncryptionParameters = CKKSEncryptionParametersBuilder::new()
		.set_poly_modulus_degree(degree)
		.set_coefficient_modulus(modulus_chain.clone())
		.build()?;

	let scale = 2f64.powi(40);

	let context = Context::new(&encryption_parameters, true, SecurityLevel::TC128)?;

	println!("Context is ready");

	// Key generation
	let keygen = KeyGenerator::new(&context)?;
	let secret_key = keygen.secret_key().clone();
	let public_key = keygen.create_public_key()?;
	let relin_keys = keygen.create_relin_keys()?;
	let galois_keys = keygen.create_galois_keys()?;

	let encryptor = Encryptor::new(&context, &public_key)?;
	let decryptor = Decryptor::new(&context, &secret_key)?;
	let evaluator = Evaluator::new(&context)?;
	let encoder = CKKSEncoder::new(&context, scale)?;

	let slot_count = encoder.get_slot_count();
	println!("Number of slots: {}", slot_count);

	// Input vector
	let mut input: Vec<f64> = Vec::with_capacity(slot_count);
	let step_size = 1.0 / (slot_count as f64 - 1.0);
	for i in 0..slot_count {
		input.push(i as f64 * step_size);
	}
	println!("Input vector: {:?}", &input[0..7]);

	println!("Evaluating polynomial PI*x^3 + 0.4x + 1 ...");

	// Create plaintexts for coefficients
	let plain_coeff3 = encoder.encode_f64(&[PI])?;
	let plain_coeff1 = encoder.encode_f64(&[0.4])?;
	let plain_coeff0 = encoder.encode_f64(&[1.0])?;

	// Encode input and encrypt
	let x_plain = encoder.encode_f64(&input)?;
	let mut x1_encrypted = encryptor.encrypt(&x_plain)?;

	// Compute x^2
	let mut x3_encrypted = evaluator.square(&x1_encrypted)?;
	evaluator
		.relinearize_inplace(&mut x3_encrypted, &relin_keys)?;
	println!(
		"Scale of x^2 before rescale: {:.2} bits",
		x3_encrypted.get_scale().log2()
	);

	// Rescale x^2
	evaluator
		.rescale_to_next_inplace(&mut x3_encrypted)?;
	println!(
		"Scale of x^2 after rescale: {:.2} bits",
		x3_encrypted.scale().log2()
	);

	// Compute and rescale PI*x
	let mut x1_encrypted_coeff3 = evaluator
        .multiply_plain(&x1_encrypted, &plain_coeff3)?;
	evaluator
		.rescale_to_next_inplace(&mut x1_encrypted_coeff3)?;
	println!(
		"Scale of PI*x after rescale: {:.2} bits",
		x1_encrypted_coeff3.scale().log2()
	);

	// Compute (PI*x)*x^2
	evaluator
		.multiply_inplace(&mut x3_encrypted, &x1_encrypted_coeff3)?;
	evaluator
		.relinearize_inplace(&mut x3_encrypted, &relin_keys)?;
	evaluator
		.rescale_to_next_inplace(&mut x3_encrypted)?;
	println!(
		"Scale of PI*x^3 after rescale: {:.2} bits",
		x3_encrypted.scale().log2()
	);

	// Compute and rescale 0.4*x
	evaluator
		.multiply_plain_inplace(&mut x1_encrypted, &plain_coeff1)?;
	evaluator
		.rescale_to_next_inplace(&mut x1_encrypted)?;
	println!(
		"Scale of 0.4*x after rescale: {:.2} bits",
		x1_encrypted.scale().log2()
	);

	// Final polynomial computation would continue here
	// TODO: https://github.com/microsoft/SEAL/blob/119dc32e135cb89c1062076a69310d4413ebc824/native/examples/5_ckks_basics.cpp#L199C5-L317C1
	Ok(())
}
