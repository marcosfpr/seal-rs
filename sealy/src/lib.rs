//!
//! # Example
//!
//! ```rust
//! use sealy::{
//!     BFVEncoder, Evaluator, BfvEncryptionParametersBuilder, CoefficientModulus, Context,
//!     Decryptor, DegreeType, Encoder, Encryptor, Evaluator, KeyGenerator, PlainModulus,
//!     SecurityLevel,
//! };
//!
//! fn main() -> anyhow::Result<()> {
//!     let params = BfvEncryptionParametersBuilder::new()
//!         .set_poly_modulus_degree(DegreeType::D8192)
//!         .set_coefficient_modulus(
//!             CoefficientModulus::create(DegreeType::D8192, &[50, 30, 30, 50, 50]).unwrap(),
//!         )
//!         .set_plain_modulus(PlainModulus::batching(DegreeType::D8192, 32)?)
//!         .build()?;
//!
//!     let ctx = Context::new(&params, false, SecurityLevel::TC128)?;
//!     let gen = KeyGenerator::new(&ctx)?;
//!
//!     let encoder = BFVEncoder::new(&ctx)?;
//!
//!     let public_key = gen.create_public_key();
//!     let secret_key = gen.secret_key();
//!
//!     let encryptor = Encryptor::with_public_key(&ctx, &public_key)?;
//!     let decryptor = Decryptor::new(&ctx, &secret_key)?;
//!     let evaluator = Evaluator::new(&ctx)?;
//!
//!     let plaintext: Vec<i64> = vec![1, 2, 3];
//!     let factor = vec![2, 2, 2];
//!
//!     let encoded_plaintext = encoder.encode(&plaintext)?;
//!     let encoded_factor = encoder.encode(&factor)?;
//!
//!     let ciphertext = encryptor.encrypt(&encoded_plaintext)?;
//!     let ciphertext_result = evaluator.multiply_plain(&ciphertext, &encoded_factor)?;
//!
//!     let decrypted = decryptor.decrypt(&ciphertext_result)?;
//!     let decoded = encoder.decode(&decrypted);
//!
//!     println!("{:?}", &decoded.into_iter().take(3).collect::<Vec<_>>()); // [2, 4, 6]
//!
//!     Ok(())
//! }
//! ```

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![warn(missing_docs)]

#[cfg(not(target_arch = "wasm32"))]
extern crate link_cplusplus;

#[allow(non_camel_case_types)]
#[allow(unused)]
mod bindgen {
	use std::os::raw::c_long;

	include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

	pub const S_OK: c_long = 0x0;
	pub const E_POINTER: c_long = 0x80004003u32 as c_long;
	pub const E_INVALIDARG: c_long = 0x80070057u32 as c_long;
	pub const E_OUTOFMEMORY: c_long = 0x8007000Eu32 as c_long;
	pub const E_UNEXPECTED: c_long = 0x8000FFFFu32 as c_long;
	pub const COR_E_IO: c_long = 0x80131620u32 as c_long;
	pub const COR_E_INVALIDOPERATION: c_long = 0x80131509u32 as c_long;
}

mod ciphertext;
mod components;
mod context;
mod decryptor;
mod encoder;
mod encryptor;
mod error;
mod evaluator;
mod ext;
mod key_generator;
mod memory;
mod modulus;
mod parameters;
mod plaintext;
mod poly_array;
mod serialization;

pub use ciphertext::Ciphertext;
pub use components::{
	marker as component_marker, Asym, AsymmetricComponents, Sym, SymAsym, SymmetricComponents,
};
pub use context::Context;
pub use decryptor::Decryptor;
pub use encoder::bfv::BFVEncoder;
pub use encoder::ckks::CKKSEncoder;
pub use encryptor::{AsymmetricEncryptor, Encryptor, SymmetricEncryptor};
pub use error::{Error, Result};
pub use evaluator::{Evaluator, EvaluatorOps};
pub use ext::tensor::{
	decryptor::TensorDecryptor, encoder::TensorEncoder, encryptor::TensorEncryptor,
	evaluator::TensorEvaluator, FromChunk, Tensor, ToChunk,
};
pub use key_generator::{GaloisKey, KeyGenerator, PublicKey, RelinearizationKey, SecretKey};
pub use memory::MemoryPool;
pub use modulus::{
	CoefficientModulusFactory, DegreeType, Modulus, PlainModulusFactory, SecurityLevel,
};
pub use parameters::*;
pub use plaintext::Plaintext;
pub use poly_array::PolynomialArray;
pub use serialization::{FromBytes, ToBytes};
