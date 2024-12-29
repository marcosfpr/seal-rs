use sealy::{
	Asym, CKKSEncoder, CKKSEncryptionParametersBuilder, Ciphertext, CoefficientModulusFactory,
	Context, Decryptor, DegreeType, EncryptionParameters, Encryptor,
	Evaluator, EvaluatorOps, KeyGenerator, Plaintext, RelinearizationKey, SchemeType, SecretKey,
	SecurityLevel,
};
use std::f64::consts::PI;

fn main() -> anyhow::Result<()> {
	println!("Example: Evaluating polynomial PI*x^3 + 0.4x + 1 using CKKS");

	let poly_modulus_degree = DegreeType::D8192;
	let modulus_chain = CoefficientModulusFactory::build(poly_modulus_degree, &[60, 40, 40, 60])?;

	let encryption_parameters: EncryptionParameters = CKKSEncryptionParametersBuilder::new()
		.set_poly_modulus_degree(poly_modulus_degree)
		.set_coefficient_modulus(modulus_chain.clone())
		.build()?;

	let scale = 2f64.powi(40);

	let context = Context::new(&encryption_parameters, true, SecurityLevel::TC128)?;

	println!("Context is ready");

	// Key generation
	let keygen = KeyGenerator::new(&context)?;
	let secret_key = keygen.secret_key().clone();
	let public_key = keygen.create_public_key();
	let relin_keys = keygen.create_relinearization_keys()?;
	let galois_keys = keygen.create_galois_keys()?;

	let encryptor = Encryptor::<Asym>::new(&context, &public_key)?;
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
	let plain_coeff3 = encoder.encode_single_f64(PI)?;
	let plain_coeff1 = encoder.encode_single_f64(0.4)?;
	let mut plain_coeff0 = encoder.encode_single_f64(1.0)?;

	// Encode input and encrypt
	let x_plain = encoder.encode_f64(&input)?;
	let mut x1_encrypted = encryptor.encrypt(&x_plain)?;

	// Compute x^2
	let mut x3_encrypted = evaluator.square(&x1_encrypted)?;
	evaluator.relinearize_inplace(&mut x3_encrypted, &relin_keys)?;
	println!(
		"Scale of x^2 before rescale: {:.2} bits",
		x3_encrypted.get_scale()?.log2()
	);

	// Rescale x^2
	evaluator.rescale_to_next_inplace(&mut x3_encrypted)?;
	println!(
		"Scale of x^2 after rescale: {:.2} bits",
		x3_encrypted.get_scale()?.log2()
	);

	// Compute and rescale PI*x
	let mut x1_encrypted_coeff3 = evaluator.multiply_plain(&x1_encrypted, &plain_coeff3)?;
	evaluator.rescale_to_next_inplace(&mut x1_encrypted_coeff3)?;
	println!(
		"Scale of PI*x after rescale: {:.2} bits",
		x1_encrypted_coeff3.get_scale()?.log2()
	);

	// Compute (PI*x)*x^2
	evaluator.multiply_inplace(&mut x3_encrypted, &x1_encrypted_coeff3)?;
	evaluator.relinearize_inplace(&mut x3_encrypted, &relin_keys)?;
	evaluator.rescale_to_next_inplace(&mut x3_encrypted)?;
	println!(
		"Scale of PI*x^3 after rescale: {:.2} bits",
		x3_encrypted.get_scale()?.log2()
	);

	// Compute and rescale 0.4*x
	evaluator.multiply_plain_inplace(&mut x1_encrypted, &plain_coeff1)?;
	evaluator.rescale_to_next_inplace(&mut x1_encrypted)?;
	println!(
		"Scale of 0.4*x after rescale: {:.2} bits",
		x1_encrypted.get_scale()?.log2()
	);


	println!(
		"The exact scales of all three terms are different:\n \
         + Exact scale in PI*x^3: {:.10}\n \
         + Exact scale in 0.4*x: {:.10}\n \
         + Exact scale in 1: {:.10}",
		x3_encrypted.get_scale()?,
		x1_encrypted.get_scale()?,
		plain_coeff0.get_scale()?
	);

	let target_parms_id = context.get_last_parms_id()?; // Lowest level available

	evaluator.mod_switch_to_inplace(&mut x3_encrypted, &target_parms_id)?;
	evaluator.mod_switch_to_inplace(&mut x1_encrypted, &target_parms_id)?;
	evaluator.mod_switch_to_inplace_plaintext(&mut plain_coeff0, &target_parms_id)?;

	println!("Normalize scales to 2^40.");
	let target_scale = 2_f64.powi(40);
	x3_encrypted.set_scale(target_scale)?;
	x1_encrypted.set_scale(target_scale)?;

	println!("Normalize encryption parameters to the lowest level.");
	let last_parms_id = x3_encrypted.get_parms_id()?;
    println!("Last parms id: {:?}", last_parms_id);
	evaluator
		.mod_switch_to_inplace(&mut x1_encrypted, &last_parms_id)?;
	evaluator
		.mod_switch_to_inplace_plaintext(&mut plain_coeff0, &last_parms_id)?;

	println!("Compute PI*x^3 + 0.4*x + 1.");

	let mut encrypted_result = evaluator.add(&x3_encrypted, &x1_encrypted)?;
	evaluator.add_plain_inplace(&mut encrypted_result, &plain_coeff0)?;

    println!("Decrypt and decode PI*x^3 + 0.4*x + 1.");
    println!("    + Expected result:");

    let mut true_result = Vec::new();
    for &x in &input {
        let value = (3.14159265 * x * x * x) + 0.4 * x + 1.0;
        true_result.push(value);
    }
    println!("True result (first 7 values): {:?}", &true_result[0..7]);

    let mut plain_result = decryptor
        .decrypt(&encrypted_result)?;

    let mut result = encoder.decode_f64(&plain_result)?;

    println!("    + Computed result ...... Correct.");
    println!("Computed result (first 7 values): {:?}", &result[0..7]);

    // Verify correctness
    for (i, (expected, computed)) in true_result.iter().zip(&result).enumerate().take(7) {
        let error = (expected - computed).abs();
        println!("Index {}: Expected = {:.10}, Computed = {:.10}, Error = {:.10}", i, expected, computed, error);
        assert!(
            error < 1e-6,
            "Result verification failed at index {}: error = {:.10}",
            i,
            error
        );
    }

	Ok(())
}
