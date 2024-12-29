use core::hash::Hash;
use std::ffi::{c_void, CString};
use std::fmt::Debug;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering};

use crate::{bindgen, serialization::CompressionType, Context, FromBytes, ToBytes};
use crate::{error::*, try_seal, MemoryPool};

use serde::ser::Error;
use serde::{Serialize, Serializer};

/// Class to store a plaintext encoded items. The data encoded for the
/// plaintext is a polynomial with coefficients modulo the plaintext modulus.
/// The degree of the plaintext polynomial must be one less than the degree of the
/// polynomial modulus. The backing array always allocates one 64-bit word
/// per each coefficient of the polynomial.
///
/// # Memory Management
/// The coefficient count of a plaintext refers to the number of word-size
/// coefficients in the plaintext, whereas its capacity refers to the number
/// of word-size coefficients that fit in the current memory allocation. In
/// high-performance applications unnecessary re-allocations should be avoided
/// by reserving enough memory for the plaintext to begin with either by
/// providing the desired capacity to the constructor as an extra argument, or
/// by calling the reserve function at any time.
///
/// When the scheme is SchemeType.BFV each coefficient of a plaintext is
/// a 64-bit word, but when the scheme is SchemeType.CKKS the plaintext is
/// by default stored in an NTT transformed form with respect to each of the
/// primes in the coefficient modulus. Thus, the size of the allocation that
/// is needed is the size of the coefficient modulus (number of primes) times
/// the degree of the polynomial modulus. In addition, a valid CKKS plaintext
/// will also store the ParmsId for the corresponding encryption parameters.
pub struct Plaintext {
	handle: AtomicPtr<c_void>,
}

impl Plaintext {
	/// Returns the handle to the underlying SEAL object.
	pub(crate) unsafe fn get_handle(&self) -> *mut c_void {
		self.handle.load(Ordering::SeqCst)
	}

	/// Constructs an empty plaintext allocating no memory.
	pub fn new() -> Result<Self> {
		let mut handle: *mut c_void = null_mut();

		try_seal!(unsafe { bindgen::Plaintext_Create1(null_mut(), &mut handle) })?;

		Ok(Self {
			handle: AtomicPtr::new(handle),
		})
	}

	/// Constructs an empty plaintext in a memory pool.
	pub fn new_with_pool(memory: &MemoryPool) -> Result<Self> {
		let mut handle: *mut c_void = null_mut();

		try_seal!(unsafe { bindgen::Plaintext_Create1(memory.get_handle(), &mut handle) })?;

		Ok(Self {
			handle: AtomicPtr::new(handle),
		})
	}

	/// Constructs a plaintext from a given hexadecimal string describing the
	/// plaintext polynomial.
	///
	/// The string description of the polynomial must adhere to the format
	/// returned by ToString(), which is of the form "7FFx^3 + 1x^1 + 3"
	/// and summarized by the following
	/// rules:
	///
	/// 1. Terms are listed in order of strictly decreasing exponent
	/// 2. Coefficient values are non-negative and in hexadecimal format (upper
	///    and lower case letters are both supported)
	/// 3. Exponents are positive and in decimal format
	/// 4. Zero coefficient terms (including the constant term) may be (but do
	///    not have to be) omitted
	/// 5. Term with the exponent value of one must be exactly written as x^1
	/// 6. Term with the exponent value of zero (the constant term) must be written
	///    as just a hexadecimal number without exponent
	/// 7. Terms must be separated by exactly \[space\]+\[space\] and minus is not
	///    allowed
	/// 8. Other than the +, no other terms should have whitespace
	///
	/// * `hex_str`: The formatted polynomial string specifying the plaintext
	///              polynomial.
	///
	/// # Panics
	/// Panics if `hex_str` contains a null character anywhere but the end of the string.
	pub fn from_hex_string(hex_str: &str) -> Result<Self> {
		let mut handle: *mut c_void = null_mut();

		let hex_string = CString::new(hex_str).unwrap();

		try_seal!(unsafe {
			bindgen::Plaintext_Create4(hex_string.as_ptr() as *mut u8, null_mut(), &mut handle)
		})?;

		Ok(Self {
			handle: AtomicPtr::new(handle),
		})
	}

	/// Gets the coefficient at the given location. Coefficients are ordered
	/// from lowest to highest degree, with the first value being the constant
	/// coefficient.
	///
	/// # Panics
	/// Panics if index is greater than len().
	pub fn get_coefficient(
		&self,
		index: usize,
	) -> u64 {
		let mut coeff: u64 = 0;

		if index > self.len() {
			panic!("Index {} out of bounds {}", index, self.len());
		}

		try_seal!(unsafe {
			bindgen::Plaintext_CoeffAt(self.get_handle(), index as u64, &mut coeff)
		})
		.expect("Fatal error in Plaintext::index().");

		coeff
	}


    /// Returns the scale of the plaintext.
    pub fn get_scale(&self) -> Result<f64> {

		let mut value: f64 = 0.0;

		try_seal!(unsafe {
			bindgen::Plaintext_Scale(self.get_handle(), &mut value)
		})?;

		Ok(value)
    }

	/// Sets the coefficient at the given location. Coefficients are ordered
	/// from lowest to highest degree, with the first value being the constant
	/// coefficient.
	///
	/// # Panics
	/// Panics if index is greater than len().
	pub fn set_coefficient(
		&mut self,
		index: usize,
		value: u64,
	) {
		if index > self.len() {
			panic!("Index {} out of bounds {}", index, self.len());
		}

		try_seal!(unsafe { bindgen::Plaintext_SetCoeffAt(self.get_handle(), index as u64, value) })
			.expect("Fatal error in Plaintext::index().");
	}


	/// Sets the number of coefficients this plaintext can hold.
	pub fn resize(
		&mut self,
		count: usize,
	) {
		try_seal!(unsafe { bindgen::Plaintext_Resize(self.get_handle(), count as u64) })
			.expect("Fatal error in Plaintext::resize().");
	}

	/// Returns the number of coefficients this plaintext can hold.
	pub fn len(&self) -> usize {
		let mut size: u64 = 0;

		try_seal!(unsafe { bindgen::Plaintext_CoeffCount(self.get_handle(), &mut size) })
			.expect("Fatal error in Plaintext::index().");

		size as usize
	}

	/// Returns `true` if the plaintext is empty.
	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}

	/// Returns whether the plaintext is in NTT form.
	pub fn is_ntt_form(&self) -> bool {
		let mut result = false;

		try_seal!(unsafe { bindgen::Plaintext_IsNTTForm(self.get_handle(), &mut result) })
			.expect("Fatal error in Plaintext::is_ntt_form().");

		result
	}
}

impl Debug for Plaintext {
	fn fmt(
		&self,
		f: &mut std::fmt::Formatter<'_>,
	) -> std::fmt::Result {
		f.debug_struct("Plaintext")
			.field("handle", &self.handle)
			.finish()
	}
}

impl Clone for Plaintext {
	fn clone(&self) -> Self {
		let mut copy = null_mut();

		try_seal!(unsafe { bindgen::Plaintext_Create5(self.get_handle(), &mut copy) })
			.expect("Internal error: Failed to copy plaintext.");

		Self {
			handle: AtomicPtr::new(copy),
		}
	}
}

impl AsRef<Plaintext> for Plaintext {
	fn as_ref(&self) -> &Self {
		self
	}
}

impl PartialEq for Plaintext {
	fn eq(
		&self,
		other: &Self,
	) -> bool {
		if self.len() == other.len() {
			for i in 0..self.len() {
				if self.get_coefficient(i) != other.get_coefficient(i) {
					return false;
				}
			}

			true
		} else {
			false
		}
	}
}

impl Hash for Plaintext {
	fn hash<H: std::hash::Hasher>(
		&self,
		state: &mut H,
	) {
		for i in 0..self.len() {
			let c = self.get_coefficient(i);
			state.write_u64(c);
		}
	}
}

impl Serialize for Plaintext {
	fn serialize<S>(
		&self,
		serializer: S,
	) -> std::result::Result<S::Ok, S::Error>
	where
		S: Serializer,
	{
		let mut num_bytes: i64 = 0;

		try_seal!(unsafe {
			bindgen::Plaintext_SaveSize(
				self.get_handle(),
				CompressionType::ZStd as u8,
				&mut num_bytes,
			)
		})
		.map_err(|e| {
			S::Error::custom(format!("Failed to get private key serialized size: {}", e))
		})?;

		let bytes = self
			.as_bytes()
			.map_err(|e| S::Error::custom(format!("Failed to serialize bytes: {}", e)))?;

		serializer.serialize_bytes(&bytes)
	}
}

impl FromBytes for Plaintext {
	type State = Context;
	/// Deserializes a byte stream into a plaintext. This requires a context, which is why
	/// Plaintext doesn't `impl Deserialize`.
	fn from_bytes(
		context: &Context,
		data: &[u8],
	) -> Result<Self> {
		let mut bytes_read = 0;

		let plaintext = Plaintext::new()?;

		try_seal!(unsafe {
			// While the interface marks data as mut, SEAL doesn't actually modify it, so we're okay.
			bindgen::Plaintext_Load(
				plaintext.get_handle(),
				context.get_handle(),
				data.as_ptr() as *mut u8,
				data.len() as u64,
				&mut bytes_read,
			)
		})?;

		Ok(plaintext)
	}
}

impl ToBytes for Plaintext {
	fn as_bytes(&self) -> Result<Vec<u8>> {
		let mut num_bytes: i64 = 0;

		try_seal!(unsafe {
			bindgen::Plaintext_SaveSize(
				self.get_handle(),
				CompressionType::ZStd as u8,
				&mut num_bytes,
			)
		})?;

		let mut data: Vec<u8> = Vec::with_capacity(num_bytes as usize);
		let mut bytes_written: i64 = 0;

		try_seal!(unsafe {
			let data_ptr = data.as_mut_ptr();

			bindgen::Plaintext_Save(
				self.get_handle(),
				data_ptr,
				num_bytes as u64,
				CompressionType::ZStd as u8,
				&mut bytes_written,
			)
		})?;

		unsafe { data.set_len(bytes_written as usize) };

		Ok(data)
	}
}

impl Drop for Plaintext {
	fn drop(&mut self) {
		try_seal!(unsafe { bindgen::Plaintext_Destroy(self.get_handle()) })
			.expect("Internal error in Plaintext::drop.");
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn can_create_and_destroy_plaintext() {
		let plaintext = Plaintext::new().unwrap();

		std::mem::drop(plaintext);
	}

	#[test]
	fn plaintext_coefficients_in_increasing_order() {
		let plaintext = Plaintext::from_hex_string("1234x^2 + 4321").unwrap();

		assert_eq!(plaintext.get_coefficient(0), 0x4321);
		assert_eq!(plaintext.get_coefficient(1), 0);
		assert_eq!(plaintext.get_coefficient(2), 0x1234);
	}
}
