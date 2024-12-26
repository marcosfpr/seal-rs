//! Sign function as a polynomial approximation

use crate::{Context, RelinearizationKey, Result};

mod ckks;
mod bfv;


/// [Evaluator] extension that allows to evaluate the sign of the ciphertext.
/// Useful for performing comparisons between ciphertexts.
pub trait SignEvaluator {
    /// The type of the ciphertext.
    type Ciphertext;

    /// Evaluates the sign of the ciphertext.
    ///
    /// # Arguments
    /// * `a` - The ciphertext to evaluate the sign of.
    fn sign_inplace(
		&self,
		a: &mut Self::Ciphertext,
        ctx: &Context,
		relin_keys: &RelinearizationKey,
	) -> Result<()>;
}


/// Coefficients for the polynomial approximation of the sign function.
pub mod coefficients {
    pub static COEFFS_N4: &[f64] = &[
        0.0,
        315.0 / 128.0,
        0.0,
        -420.0 / 128.0,
        0.0,
        378.0 / 128.0,
        0.0,
        -180.0 / 128.0,
        0.0,
        35.0 / 128.0,
    ];

    pub static COEFFS_N1: &[f64] = &[
        0.0,
        3.0 / 2.0,
        0.0,
        -1.0 / 2.0,
    ];

    pub static COEFFS_N3: &[f64] = &[
        0.0,
        35.0 / 16.0,
        0.0,
        -35.0 / 16.0,
        0.0,
        21.0 / 16.0,
        0.0,
        -5.0 / 16.0,
    ];

    pub static COEFFS_N7: &[f64] = &[
        0.0,
        6435.0 / 2048.0,
        0.0,
        -15015.0 / 2048.0,
        0.0,
        27027.0 / 2048.0,
        0.0,
        -32175.0 / 2048.0,
        0.0,
        25025.0 / 2048.0,
        0.0,
        -12285.0 / 2048.0,
        0.0,
        3465.0 / 2048.0,
        0.0,
        -429.0 / 2048.0,
    ];

    pub static COEFFS_N15: &[f64] = &[
        0.0,
        300540195.0 / 67108864.0,
        0.0,
        -1502700975.0 / 67108864.0,
        0.0,
        6311344095.0 / 67108864.0,
        0.0,
        -19535112675.0 / 67108864.0,
        0.0,
        45581929575.0 / 67108864.0,
        0.0,
        -82047473235.0 / 67108864.0,
        0.0,
        115707975075.0 / 67108864.0,
        0.0,
        -128931743655.0 / 67108864.0,
        0.0,
        113763303225.0 / 67108864.0,
        0.0,
        -79168614525.0 / 67108864.0,
        0.0,
        42977247885.0 / 67108864.0,
        0.0,
        -17836407225.0 / 67108864.0,
        0.0,
        5469831549.0 / 67108864.0,
        0.0,
        -1168767425.0 / 67108864.0,
        0.0,
        155451825.0 / 67108864.0,
        0.0,
        -9694845.0 / 67108864.0,
    ];
}
