use std::ffi::c_void;
use std::fmt::Debug;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering};

use crate::error::Result;
use crate::{bindgen, try_seal, Context, MemoryPool, Plaintext};

/// To create CKKS plaintexts we need a special encoder: there is no other way
/// to create them. The BatchEncoder cannot be used with the
/// CKKS scheme. The CKKSEncoder encodes vectors of real or complex numbers into
/// Plaintext objects, which can subsequently be encrypted. At a high level this
/// looks a lot like what BatchEncoder does for the BFV scheme, but the theory
/// behind it is completely different.
pub struct CKKSEncoder {
	handle: AtomicPtr<c_void>,
	parms_id: Vec<u64>,
	scale: f64,
}

impl CKKSEncoder {
	/// Creates a CKKSEncoder. It is necessary that the encryption parameters
	/// given through the SEALContext object support it.
	///
	/// * `ctx` - The Context
	/// * `scale` - The scaling factor
	pub fn new(
		ctx: &Context,
		scale: f64,
	) -> Result<Self> {
		let mut handle: *mut c_void = null_mut();

		// TODO: Investigate how to properly set the parms_id in the ckks encoding.
		let parms_id = ctx.get_first_parms_id()?;

		try_seal!(unsafe { bindgen::CKKSEncoder_Create(ctx.get_handle(), &mut handle) })?;

		Ok(Self {
			handle: AtomicPtr::new(handle),
			parms_id,
			scale,
		})
	}

	/// Get the handle to the underlying SEAL object.
	pub(crate) unsafe fn get_handle(&self) -> *mut c_void {
		self.handle.load(Ordering::SeqCst)
	}

	/// Returns the number of  slots in this encoder produces.
	pub fn get_slot_count(&self) -> usize {
		let mut count: u64 = 0;

		try_seal!(unsafe { bindgen::CKKSEncoder_SlotCount(self.get_handle(), &mut count) })
			.expect("Internal error in BVTEncoder::get_slot_count().");

		count as usize
	}

	/// Creates a plaintext from a given matrix of f64 data.
	///
	/// The floating-point coefficients of `data`
	/// will be scaled up by the parameter `scale'. This is necessary since even in
	/// the CKKS scheme the plaintext elements are fundamentally polynomials with
	/// integer coefficients. It is instructive to think of the scale as determining
	/// the bit-precision of the encoding; naturally it will affect the precision of
	/// the result.
	/// In CKKS the message is stored modulo coeff_modulus (in BFV it is stored modulo
	/// plain_modulus), so the scaled message must not get too close to the total size
	/// of coeff_modulus. In this case our coeff_modulus is quite large (200 bits) so
	/// we have little to worry about in this regard. For this simple example a 30-bit
	/// scale is more than enough.
	///
	///  * `data` - The `2xN` matrix of integers modulo plaintext modulus to batch
	pub fn encode_f64(
		&self,
		data: &[f64],
	) -> Result<Plaintext> {
		let mem = MemoryPool::new()?;

		let plaintext = Plaintext::new()?;

		// I pinky promise SEAL won't mutate data, the C bindings just aren't
		// const correct.
		try_seal!(unsafe {
			let mut parms_id = self.parms_id.clone();
			let parms_id_ptr = parms_id.as_mut_ptr();
			bindgen::CKKSEncoder_Encode1(
				self.get_handle(),
				data.len() as u64,
				data.as_ptr() as *mut f64,
				parms_id_ptr,
				self.scale,
				plaintext.get_handle(),
				mem.get_handle(),
			)
		})?;

		Ok(plaintext)
	}

	/// Creates a plaintext from a given f64 value.
	///
	/// The floating-point coefficients of `data`
	/// will be scaled up by the parameter `scale'. This is necessary since even in
	/// the CKKS scheme the plaintext elements are fundamentally polynomials with
	/// integer coefficients. It is instructive to think of the scale as determining
	/// the bit-precision of the encoding; naturally it will affect the precision of
	/// the result.
	/// In CKKS the message is stored modulo coeff_modulus (in BFV it is stored modulo
	/// plain_modulus), so the scaled message must not get too close to the total size
	/// of coeff_modulus. In this case our coeff_modulus is quite large (200 bits) so
	/// we have little to worry about in this regard. For this simple example a 30-bit
	/// scale is more than enough.
	///
    /// * `value` - The f64 value to encode
    pub fn encode_single_f64(
        &self,
        value: f64,
    ) -> Result<Plaintext> {
        let mem = MemoryPool::new()?;

        let plaintext = Plaintext::new()?;

        // I pinky promise SEAL won't mutate data, the C bindings just aren't
        // const correct.
        try_seal!(unsafe {
            let mut parms_id = self.parms_id.clone();
            let parms_id_ptr = parms_id.as_mut_ptr();
            bindgen::CKKSEncoder_Encode3(
                self.get_handle(),
                value,
                parms_id_ptr,
                self.scale,
                plaintext.get_handle(),
                mem.get_handle(),
            )
        })?;

        Ok(plaintext)
    }


	/// Inverse of encode. This function decodes a given plaintext into
	/// a list of f64 elements.
	///
	///  * `plaintext` - The plaintext polynomial to unbatch
	pub fn decode_f64(
		&self,
		plaintext: &Plaintext,
	) -> Result<Vec<f64>> {
		let mut data = Vec::with_capacity(self.get_slot_count());
		let data_ptr = data.as_mut_ptr();
		let mut size: u64 = 0;

		// I pinky promise SEAL won't mutate data, the C bindings just aren't
		// const correct.
		try_seal!(unsafe {
			bindgen::CKKSEncoder_Decode1(
				self.get_handle(),
				plaintext.get_handle(),
				&mut size,
				data_ptr,
				null_mut(),
			)
		})?;

		if data.capacity() < size as usize {
			panic!("Allocation overflow BVTEncoder::decode_unsigned");
		}

		unsafe {
			data.set_len(size as usize);
		}

		Ok(data)
	}
}

impl Debug for CKKSEncoder {
	fn fmt(
		&self,
		f: &mut std::fmt::Formatter<'_>,
	) -> std::fmt::Result {
		f.debug_struct("CKKSEncoder")
			.field("handle", &self.handle)
			.field("parms_id", &self.parms_id)
			.field("scale", &self.scale)
			.finish()
	}
}

impl Drop for CKKSEncoder {
	fn drop(&mut self) {
		unsafe {
			bindgen::CKKSEncoder_Destroy(self.get_handle());
		}
	}
}

#[cfg(test)]
mod tests {
	use crate::{
		CKKSEncoder, CKKSEncryptionParametersBuilder, CoefficientModulusFactory, Context,
		DegreeType, EncryptionParameters, Error, SecurityLevel,
	};

	fn float_assert_eq(
		a: f64,
		b: f64,
	) {
		assert!((a - b).abs() < 0.0001);
	}

	fn float_iter_assert_eq(
		a: impl IntoIterator<Item = f64>,
		b: impl IntoIterator<Item = f64>,
	) {
		for (a, b) in a.into_iter().zip(b.into_iter()) {
			float_assert_eq(a, b);
		}
	}

	fn create_ckks_context(
		degree: DegreeType,
		bit_sizes: &[i32],
	) -> Result<Context, Error> {
		let security_level = SecurityLevel::TC128;
		let expand_mod_chain = false;
		let modulus_chain = CoefficientModulusFactory::build(degree, bit_sizes)?;
		let encryption_parameters: EncryptionParameters = CKKSEncryptionParametersBuilder::new()
			.set_poly_modulus_degree(degree)
			.set_coefficient_modulus(modulus_chain)
			.build()?;

		let ctx = Context::new(&encryption_parameters, expand_mod_chain, security_level)?;

		Ok(ctx)
	}

	/// Test to ensure that the CKKS encoder can be created and destroyed without issues.
	#[test]
	fn can_create_and_drop_ckks_encoder() {
		let ctx = create_ckks_context(DegreeType::D8192, &[60, 40, 40, 60]).unwrap();

		// Create CKKS encoder
		let encoder = CKKSEncoder::new(&ctx, 2.0f64.powi(40)).unwrap();

		// Drop the encoder to ensure cleanup works without errors
		std::mem::drop(encoder);
	}

	/// Test to ensure the encoder correctly retrieves the slot count for CKKS.
	#[test]
	fn can_get_slots_ckks_encoder() {
		let ctx = create_ckks_context(DegreeType::D8192, &[60, 40, 40, 60]).unwrap();

		let encoder = CKKSEncoder::new(&ctx, 2.0f64.powi(40)).unwrap();

		// Slot count for CKKS encoder should be half of the polynomial degree (N/2)
		assert_eq!(encoder.get_slot_count(), 8192 / 2);
	}

	/// Test encoding and decoding of a vector of unsigned floats in CKKS.
	#[test]
	fn can_get_encode_and_decode_unsigned() {
		let ctx = create_ckks_context(DegreeType::D8192, &[60, 40, 40, 60]).unwrap();

		let encoder = CKKSEncoder::new(&ctx, 2.0f64.powi(40)).unwrap();

		let mut data = Vec::with_capacity(4096);

		// Fill vector with data (e.g., numbers from 0 to slot count as f64)
		for i in 0..encoder.get_slot_count() {
			data.push(i as f64);
		}

		// Encode and decode the data
		let plaintext = encoder.encode_f64(&data).unwrap();
		let data_decoded: Vec<f64> = encoder.decode_f64(&plaintext).unwrap();

		// Assert that the original data and the decoded data are equal
		float_iter_assert_eq(data, data_decoded);
	}

	/// Test encoding and decoding of a vector of signed floats in CKKS.
	#[test]
	fn can_get_encode_and_decode_signed() {
		let ctx = create_ckks_context(DegreeType::D8192, &[60, 40, 40, 60]).unwrap();

		let encoder = CKKSEncoder::new(&ctx, 2.0f64.powi(40)).unwrap();

		let mut data = Vec::with_capacity(4096);

		// Fill vector with negative and positive floats
		for i in 0..encoder.get_slot_count() {
			data.push(i as f64 - 2048.0);
		}

		// Encode and decode the data
		let plaintext = encoder.encode_f64(&data).unwrap();
		let data_decoded: Vec<f64> = encoder.decode_f64(&plaintext).unwrap();

		// Assert that the original data and the decoded data are equal
		float_iter_assert_eq(data, data_decoded);
	}

	/// Test encoding and decoding of a scalar (single) signed float in CKKS.
	#[test]
	fn scalar_encoder_can_encode_decode_signed() {
		let ctx = create_ckks_context(DegreeType::D8192, &[60, 40, 40, 60]).unwrap();

		let encoder = CKKSEncoder::new(&ctx, 2.0f64.powi(40)).unwrap();

		// Single value to encode and decode
		let encoded = encoder.encode_f64(&[-15.5f64]).unwrap();
		let decoded: Vec<f64> = encoder.decode_f64(&encoded).unwrap();

		// Assert that the decoded value matches the original
		float_assert_eq(decoded[0], -15.5);
	}

	/// Test encoding and decoding of a scalar (single) unsigned float in CKKS.
	#[test]
	fn scalar_encoder_can_encode_decode_unsigned() {
		let ctx = create_ckks_context(DegreeType::D8192, &[60, 40, 40, 60]).unwrap();

		let encoder = CKKSEncoder::new(&ctx, 2.0f64.powi(40)).unwrap();

		// Single positive value
		let encoded = encoder.encode_f64(&[42.0f64]).unwrap();
		let decoded: Vec<f64> = encoder.decode_f64(&encoded).unwrap();

		// Assert that the decoded value matches the original
		float_assert_eq(decoded[0], 42.0);
	}

	/// Test encoding and decoding a float vector. CKKS handles floating-point numbers, so this is expected to work.
	#[test]
	fn can_get_encode_and_decode_float() {
		let ctx = create_ckks_context(DegreeType::D8192, &[60, 40, 40, 60]).unwrap();

		let encoder = CKKSEncoder::new(&ctx, 2.0f64.powi(40)).unwrap();

		let data: Vec<f64> = (0..encoder.get_slot_count())
			.map(|i| (i as f64) / 2.0)
			.collect();

		// Encode and decode the float vector
		let plaintext = encoder.encode_f64(&data).unwrap();
		let decoded_data: Vec<f64> = encoder.decode_f64(&plaintext).unwrap();

		// Assert that the original and decoded data match within a small tolerance
		float_iter_assert_eq(data, decoded_data);
	}
}
