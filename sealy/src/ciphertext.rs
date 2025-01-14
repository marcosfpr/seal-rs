use std::ffi::c_void;
use std::fmt::Debug;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering};

use crate::{bindgen, serialization::CompressionType, Context, FromBytes, ToBytes};
use crate::{error::*, try_seal, EncryptionParameters};

/// Class to store a ciphertext element. The data for a ciphertext consists
/// of two or more polynomials, which are in Microsoft SEAL stored in a CRT
/// form with respect to the factors of the coefficient modulus. This data
/// itself is not meant to be modified directly by the user, but is instead
/// operated on by functions in the Evaluator class. The size of the backing
/// array of a ciphertext depends on the encryption parameters and the size
/// of the ciphertext (at least 2). If the PolyModulusDegree encryption
/// parameter is N, and the number of primes in the CoeffModulus encryption
/// parameter is K, then the ciphertext backing array requires precisely
/// 8*N*K*size bytes of memory. A ciphertext also carries with it the
/// parmsId of its associated encryption parameters, which is used to check
/// the validity of the ciphertext for homomorphic operations and decryption.
///
/// # Memory Management
/// The size of a ciphertext refers to the number of polynomials it contains,
/// whereas its capacity refers to the number of polynomials that fit in the
/// current memory allocation. In high-performance applications unnecessary
/// re-allocations should be avoided by reserving enough memory for the
/// ciphertext to begin with either by providing the desired capacity to the
/// constructor as an extra argument, or by calling the reserve function at
/// any time.
pub struct Ciphertext {
	handle: AtomicPtr<c_void>,
}

impl Ciphertext {
	/// Creates a new empty plaintext. Use an encoder to populate with a value.
	pub fn new() -> Result<Self> {
		let mut handle: *mut c_void = null_mut();

		try_seal!(unsafe { bindgen::Ciphertext_Create1(null_mut(), &mut handle) })?;

		Ok(Self {
			handle: AtomicPtr::new(handle),
		})
	}

	/// Returns the handle to the underlying SEAL object.
	pub(crate) unsafe fn get_handle(&self) -> *mut c_void {
		self.handle.load(Ordering::SeqCst)
	}

	/// Returns the number of polynomials in this ciphertext.
	pub fn num_polynomials(&self) -> u64 {
		let mut size: u64 = 0;

		try_seal!(unsafe { bindgen::Ciphertext_Size(self.get_handle(), &mut size) }).unwrap();

		size
	}

	/// Returns the number of components in the coefficient modulus.
	pub fn coeff_modulus_size(&self) -> u64 {
		let mut size: u64 = 0;

		try_seal!(unsafe { bindgen::Ciphertext_CoeffModulusSize(self.get_handle(), &mut size) })
			.unwrap();

		size
	}

	/// Returns the value at a specific point in the coefficient array. This is
	/// not publically exported as it leaks the encoding of the array.
	#[allow(dead_code)]
	pub(crate) fn get_data(
		&self,
		index: usize,
	) -> Result<u64> {
		let mut value: u64 = 0;

		try_seal!(unsafe {
			bindgen::Ciphertext_GetDataAt1(self.get_handle(), index as u64, &mut value)
		})?;

		Ok(value)
	}

	/// Returns the coefficient in the form the ciphertext is currently in (NTT
	/// form or not). For BFV, this will be the coefficient in the residual
	/// number system (RNS) format.
	pub fn get_coefficient(
		&self,
		poly_index: usize,
		coeff_index: usize,
	) -> Result<Vec<u64>> {
		let size = self.coeff_modulus_size();
		let mut data: Vec<u64> = Vec::with_capacity(size as usize);

		try_seal!(unsafe {
			let data_ptr = data.as_mut_ptr();

			bindgen::Ciphertext_GetDataAt2(
				self.get_handle(),
				poly_index as u64,
				coeff_index as u64,
				data_ptr,
			)
		})?;

		unsafe { data.set_len(size as usize) };

		Ok(data.clone())
	}

	/// Returns whether the ciphertext is in NTT form.
	pub fn is_ntt_form(&self) -> bool {
		let mut result = false;

		try_seal!(unsafe { bindgen::Ciphertext_IsNTTForm(self.get_handle(), &mut result) })
			.expect("Fatal error in Plaintext::is_ntt_form().");

		result
	}

	/// Returns the scale of the ciphertext.
	pub fn get_scale(&self) -> Result<f64> {
		let mut value: f64 = 0.0;

		try_seal!(unsafe { bindgen::Ciphertext_Scale(self.get_handle(), &mut value) })?;

		Ok(value)
	}

	/// Sets the scale of the ciphertext.
	pub fn set_scale(
		&self,
		scale: f64,
	) -> Result<()> {
		try_seal!(unsafe { bindgen::Ciphertext_SetScale(self.get_handle(), scale) })?;
		Ok(())
	}

	/// Returns the parms_id of the ciphertext.
	pub fn get_parms_id(&self) -> Result<Vec<u64>> {
		let mut parms_id: Vec<u64> =
			Vec::with_capacity(EncryptionParameters::block_size() as usize);
		try_seal!(unsafe {
			let parms_id_ptr = parms_id.as_mut_ptr();
			bindgen::Ciphertext_ParmsId(self.get_handle(), parms_id_ptr)
		})?;
		unsafe { parms_id.set_len(EncryptionParameters::block_size() as usize) };
		Ok(parms_id)
	}
}

impl Debug for Ciphertext {
	fn fmt(
		&self,
		f: &mut std::fmt::Formatter<'_>,
	) -> std::fmt::Result {
		f.debug_struct("Ciphertext")
			.field("handle", &self.handle)
			.finish()
	}
}

impl Clone for Ciphertext {
	fn clone(&self) -> Self {
		let mut handle = null_mut();

		try_seal!(unsafe { bindgen::Ciphertext_Create2(self.get_handle(), &mut handle) })
			.expect("Fatal error: Failed to clone ciphertext");

		Self {
			handle: AtomicPtr::new(handle),
		}
	}
}

impl AsRef<Ciphertext> for Ciphertext {
	fn as_ref(&self) -> &Self {
		self
	}
}

impl PartialEq for Ciphertext {
	fn eq(
		&self,
		other: &Self,
	) -> bool {
		self.as_bytes() == other.as_bytes()
	}
}

impl ToBytes for Ciphertext {
	fn as_bytes(&self) -> Result<Vec<u8>> {
		let mut num_bytes: i64 = 0;

		try_seal!(unsafe {
			bindgen::Ciphertext_SaveSize(
				self.get_handle(),
				CompressionType::ZStd as u8,
				&mut num_bytes,
			)
		})?;

		let mut data: Vec<u8> = Vec::with_capacity(num_bytes as usize);
		let mut bytes_written: i64 = 0;

		try_seal!(unsafe {
			let data_ptr = data.as_mut_ptr();

			bindgen::Ciphertext_Save(
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

impl FromBytes for Ciphertext {
	type State = Context;
	fn from_bytes(
		context: &Context,
		bytes: &[u8],
	) -> Result<Self> {
		let ciphertext = Self::new()?;
		let mut bytes_read = 0i64;

		try_seal!(unsafe {
			bindgen::Ciphertext_Load(
				ciphertext.get_handle(),
				context.get_handle(),
				bytes.as_ptr() as *mut u8,
				bytes.len() as u64,
				&mut bytes_read,
			)
		})?;

		Ok(ciphertext)
	}
}

impl Drop for Ciphertext {
	fn drop(&mut self) {
		try_seal!(unsafe { bindgen::Ciphertext_Destroy(self.get_handle()) })
			.expect("Internal error in Ciphertext::drop");
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn can_create_and_destroy_ciphertext() {
		let ciphertext = Ciphertext::new().unwrap();

		std::mem::drop(ciphertext);
	}
}
