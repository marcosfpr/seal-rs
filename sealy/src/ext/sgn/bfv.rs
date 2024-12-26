use crate::{Evaluator, Ciphertext, Context, RelinearizationKey};

use super::SignEvaluator;

impl SignEvaluator for Evaluator {
	type Ciphertext = Ciphertext;

	fn sign_inplace(
		&self,
		a: &mut Self::Ciphertext,
		ctx: &Context,
		relin_keys: &RelinearizationKey,
	) -> crate::Result<()> {
		todo!()
	}
}
