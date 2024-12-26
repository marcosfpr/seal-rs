use crate::{CKKSEncoder, Evaluator, Ciphertext, Context, Evaluator, RelinearizationKey};

use super::{coefficients::{COEFFS_N15, COEFFS_N3, COEFFS_N7}, SignEvaluator};

impl SignEvaluator for Evaluator {
	type Ciphertext = Ciphertext;

	fn sign_inplace(
		&self,
		a: &mut Self::Ciphertext,
		ctx: &Context,
		relin_keys: &RelinearizationKey,
	) -> crate::Result<()> {

        let evaluator = Evaluator::new(ctx)?;

        let scale = 2.0_f64.powi(40);
        let encoder = CKKSEncoder::new(ctx, scale)?;


        let plain_poly_3 = encoder.encode_f64(&COEFFS_N3)?;
        let plain_poly_7 = encoder.encode_f64(&COEFFS_N7)?;
        let plain_poly_15 = encoder.encode_f64(&COEFFS_N15)?;

        evaluator.multiply_plain_inplace(a, &plain_poly_3)?;
        evaluator.relinearize_inplace(a, relin_keys)?;
        // TODO: rescale to next inplace

        evaluator.multiply_plain_inplace(a, &plain_poly_7)?;
        evaluator.relinearize_inplace(a, relin_keys)?;
        // TODO: rescale to next inplace
        

        for i in 0..2 {
            evaluator.multiply_plain_inplace(a, &plain_poly_15)?;
            evaluator.relinearize_inplace(a, relin_keys)?;
            // TODO: rescale to next inplace
        }

	    Ok(())
}
}
