use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::atomic::AtomicPtr;
use std::sync::atomic::Ordering;

use crate::bindgen;
use crate::error::*;
use crate::try_seal;
use crate::{Ciphertext, Context, GaloisKey, Plaintext, RelinearizationKey};

/// Provides operations on ciphertexts. Due to the properties of the encryption scheme, the arithmetic operations
/// pass through the encryption layer to the underlying plaintext, changing it according to the type of the
/// operation. Since the plaintext elements are fundamentally polynomials in the polynomial quotient ring
/// Z_T[x]/(X^N+1), where T is the plaintext modulus and X^N+1 is the polynomial modulus, this is the ring where
/// the arithmetic operations will take place. BatchEncoder (batching) provider an alternative possibly more
/// convenient view of the plaintext elements as 2-by-(N2/2) matrices of integers modulo the plaintext modulus. In
/// the batching view the arithmetic operations act on the matrices element-wise. Some of the operations only apply
/// in the batching view, such as matrix row and column rotations. Other operations such as relinearization have no
/// semantic meaning but are necessary for performance reasons.
///
/// # Arithmetic Operations
/// The core operations are arithmetic operations, in particular multiplication and addition of ciphertexts. In
/// addition to these, we also provide negation, subtraction, squaring, exponentiation, and multiplication and
/// addition of several ciphertexts for convenience. in many cases some of the inputs to a computation are plaintext
/// elements rather than ciphertexts. For this we provide fast "plain" operations: plain addition, plain
/// subtraction, and plain multiplication.
///
/// # Relinearization
/// One of the most important non-arithmetic operations is relinearization, which takes as input a ciphertext of
/// size K+1 and relinearization keys (at least K-1 keys are needed), and changes the size of the ciphertext down
/// to 2 (minimum size). For most use-cases only one relinearization key suffices, in which case relinearization
/// should be performed after every multiplication. Homomorphic multiplication of ciphertexts of size K+1 and L+1
/// outputs a ciphertext of size K+L+1, and the computational cost of multiplication is proportional to K*L. Plain
/// multiplication and addition operations of any type do not change the size. Relinearization requires
/// relinearization keys to have been generated.
///
/// # Rotations
/// When batching is enabled, we provide operations for rotating the plaintext matrix rows cyclically left or right,
/// and for rotating the columns (swapping the rows). Rotations require Galois keys to have been generated.
///
/// # Other Operations
/// We also provide operations for transforming ciphertexts to NTT form and back, and for transforming plaintext
/// polynomials to NTT form. These can be used in a very fast plain multiplication variant, that assumes the inputs
/// to be in NTT form. Since the NTT has to be done in any case in plain multiplication, this function can be used
/// when e.g. one plaintext input is used in several plain multiplication, and transforming it several times would
/// not make sense.
///
/// # NTT form
/// When using the BFV scheme (SchemeType.BFV), all plaintexts and ciphertexts should remain by default in the usual
/// coefficient representation, i.e., not in NTT form. When using the CKKS scheme (SchemeType.CKKS), all plaintexts
/// and ciphertexts should remain by default in NTT form. We call these scheme-specific NTT states the "default NTT
/// form". Some functions, such as add, work even if the inputs are not in the default state, but others, such as
/// multiply, will throw an exception. The output of all evaluation functions will be in the same state as the
/// input(s), with the exception of the TransformToNTT and TransformFromNTT functions, which change the state.
/// Ideally, unless these two functions are called, all other functions should "just work".
pub struct Evaluator {
	handle: AtomicPtr<c_void>,
}

/// Operations provided by the evaluator.
pub trait EvaluatorOps {
    /// The plaintext type.
    type Plaintext;
    /// The ciphertext type.
    type Ciphertext;

	/// Negates a ciphertext inplace.
	///   * `a` - the value to negate
	fn negate_inplace(
		&self,
		a: &mut Self::Ciphertext,
	) -> Result<()>;

	/// Negates a ciphertext into a new ciphertext.
	///   * `a` - the value to negate
	fn negate(
		&self,
		a: &Self::Ciphertext,
	) -> Result<Self::Ciphertext>;

	/// Add `a` and `b` and store the result in `a`.
	///  * `a` - the accumulator
	///  * `b` - the added value
	fn add_inplace(
		&self,
		a: &mut Self::Ciphertext,
		b: &Self::Ciphertext,
	) -> Result<()>;

	/// Adds `a` and `b`.
	///  * `a` - first operand
	///  * `b` - second operand
	fn add(
		&self,
		a: &Self::Ciphertext,
		b: &Self::Ciphertext,
	) -> Result<Self::Ciphertext>;

	/// Performs an addition reduction of multiple ciphertexts packed into a slice.
	///  * `a` - a slice of ciphertexts to sum.
	fn add_many(
		&self,
		a: &[Self::Ciphertext],
	) -> Result<Self::Ciphertext>;

	/// Performs an multiplication reduction of multiple ciphertexts packed into a slice. This
	///  method creates a tree of multiplications with relinearization after each operation.
	///  * `a` - a slice of ciphertexts to sum.
	///  * `relin_keys` - the relinearization keys.
	fn multiply_many(
		&self,
		a: &[Self::Ciphertext],
		relin_keys: &RelinearizationKey,
	) -> Result<Self::Ciphertext>;

	/// Subtracts `b` from `a` and stores the result in `a`.
	///  * `a` - the left operand and destination
	///  * `b` - the right operand
	fn sub_inplace(
		&self,
		a: &mut Self::Ciphertext,
		b: &Self::Ciphertext,
	) -> Result<()>;

	/// Subtracts `b` from `a`.
	///  * `a` - the left operand
	///  * `b` - the right operand
	fn sub(
		&self,
		a: &Self::Ciphertext,
		b: &Self::Ciphertext,
	) -> Result<Self::Ciphertext>;

	/// Multiplies `a` and `b` and stores the result in `a`.
	///  * `a` - the left operand and destination.
	///  * `b` - the right operand.
	fn multiply_inplace(
		&self,
		a: &mut Self::Ciphertext,
		b: &Self::Ciphertext,
	) -> Result<()>;

	/// Multiplies `a` and `b`.
	///  * `a` - the left operand.
	///  * `b` - the right operand.
	fn multiply(
		&self,
		a: &Self::Ciphertext,
		b: &Self::Ciphertext,
	) -> Result<Self::Ciphertext>;

	/// Squares `a` and stores the result in `a`.
	///  * `a` - the value to square.
	fn square_inplace(
		&self,
		a: &mut Self::Ciphertext,
	) -> Result<()>;

	/// Squares `a`.
	///  * `a` - the value to square.
	fn square(
		&self,
		a: &Self::Ciphertext,
	) -> Result<Self::Ciphertext>;

	/// Given a ciphertext encrypted modulo q_1...q_k, this function switches the modulus down to q_1...q_{k-1} and
	/// stores the result in the destination parameter.
	///
	/// # Remarks
	/// In the BFV scheme if you've set up a coefficient modulus chain, this reduces the
	/// number of bits needed to represent the ciphertext. This in turn speeds up operations.
	///
	/// If you haven't set up a modulus chain, don't use this.
	///
	/// TODO: what does this mean for CKKS?
	fn mod_switch_to_next(
		&self,
		a: &Self::Ciphertext,
	) -> Result<Self::Ciphertext>;

	/// Given a ciphertext encrypted modulo q_1...q_k, this function switches the modulus down to q_1...q_{k-1} and
	/// stores the result in the destination parameter. This does function does so in-place.
	///
	/// # Remarks
	/// In the BFV scheme if you've set up a coefficient modulus chain, this reduces the
	/// number of bits needed to represent the ciphertext. This in turn speeds up operations.
	///
	/// If you haven't set up a modulus chain, don't use this.
	///
	/// TODO: what does this mean for CKKS?
	fn mod_switch_to_next_inplace(
		&self,
		a: &Self::Ciphertext,
	) -> Result<()>;

	/// Modulus switches an NTT transformed plaintext from modulo q_1...q_k down to modulo q_1...q_{k-1}.
	fn mod_switch_to_next_plaintext(
		&self,
		a: &Self::Plaintext,
	) -> Result<Self::Plaintext>;

	/// Modulus switches an NTT transformed plaintext from modulo q_1...q_k down to modulo q_1...q_{k-1}.
	/// This variant does so in-place.
	fn mod_switch_to_next_inplace_plaintext(
		&self,
		a: &Self::Plaintext,
	) -> Result<()>;

	/// This functions raises encrypted to a power and stores the result in the destination parameter. Dynamic
	/// memory allocations in the process are allocated from the memory pool pointed to by the given
	/// MemoryPoolHandle. The exponentiation is done in a depth-optimal order, and relinearization is performed
	/// automatically after every multiplication in the process. In relinearization the given relinearization keys
	/// are used.
	fn exponentiate(
		&self,
		a: &Self::Ciphertext,
		exponent: u64,
		relin_keys: &RelinearizationKey,
	) -> Result<Self::Ciphertext>;

	/// This functions raises encrypted to a power and stores the result in the destination parameter. Dynamic
	/// memory allocations in the process are allocated from the memory pool pointed to by the given
	/// MemoryPoolHandle. The exponentiation is done in a depth-optimal order, and relinearization is performed
	/// automatically after every multiplication in the process. In relinearization the given relinearization keys
	/// are used.
	fn exponentiate_inplace(
		&self,
		a: &Self::Ciphertext,
		exponent: u64,
		relin_keys: &RelinearizationKey,
	) -> Result<()>;

	/// Adds a ciphertext and a plaintext.
	/// * `a` - the ciphertext
	/// * `b` - the plaintext
	fn add_plain(
		&self,
		a: &Self::Ciphertext,
		b: &Self::Plaintext,
	) -> Result<Self::Ciphertext>;

	/// Adds a ciphertext and a plaintext.
	/// * `a` - the ciphertext
	/// * `b` - the plaintext
	fn add_plain_inplace(
		&self,
		a: &mut Self::Ciphertext,
		b: &Self::Plaintext,
	) -> Result<()>;

	/// Subtract a plaintext from a ciphertext.
	/// * `a` - the ciphertext
	/// * `b` - the plaintext
	fn sub_plain(
		&self,
		a: &Self::Ciphertext,
		b: &Self::Plaintext,
	) -> Result<Self::Ciphertext>;

	/// Subtract a plaintext from a ciphertext and store the result in the ciphertext.
	///  * `a` - the ciphertext
	///  * `b` - the plaintext
	fn sub_plain_inplace(
		&self,
		a: &mut Self::Ciphertext,
		b: &Self::Plaintext,
	) -> Result<()>;

	/// Multiply a ciphertext by a plaintext.
	///  * `a` - the ciphertext
	///  * `b` - the plaintext
	fn multiply_plain(
		&self,
		a: &Self::Ciphertext,
		b: &Self::Plaintext,
	) -> Result<Self::Ciphertext>;


	/// Multiply a ciphertext by a plaintext and store in the ciphertext.
	///  * `a` - the ciphertext
	///  * `b` - the plaintext
	fn multiply_plain_inplace(
		&self,
		a: &mut Self::Ciphertext,
		b: &Self::Plaintext,
	) -> Result<()>;

	/// This functions relinearizes a ciphertext in-place, reducing it to 2 polynomials. This
	/// reduces future noise growth under multiplication operations.
	fn relinearize_inplace(
		&self,
		a: &mut Self::Ciphertext,
		relin_keys: &RelinearizationKey,
	) -> Result<()>;

	/// This functions relinearizes a ciphertext, reducing it to 2 polynomials. This
	/// reduces future noise growth under multiplication operations.
	fn relinearize(
		&self,
		a: &Self::Ciphertext,
		relin_keys: &RelinearizationKey,
	) -> Result<Self::Ciphertext>;

	/// Rotates plaintext matrix rows cyclically.
	///
	/// When batching is used with the BFV scheme, this function rotates the encrypted plaintext matrix rows
	/// cyclically to the left (steps > 0) or to the right (steps < 0). Since the size of the batched matrix
	/// is 2-by-(N/2), where N is the degree of the polynomial modulus, the number of steps to rotate must have
	/// absolute value at most N/2-1.
	///
	/// * `a` - The ciphertext to rotate
	/// * `steps` - The number of steps to rotate (positive left, negative right)
	/// * `galois_keys` - The Galois keys
	fn rotate_rows(
		&self,
		a: &Self::Ciphertext,
		steps: i32,
		galois_keys: &GaloisKey,
	) -> Result<Self::Ciphertext>;

	/// Rotates plaintext matrix rows cyclically. This variant does so in-place
	///
	/// When batching is used with the BFV scheme, this function rotates the encrypted plaintext matrix rows
	/// cyclically to the left (steps &gt; 0) or to the right (steps &lt; 0). Since the size of the batched matrix
	/// is 2-by-(N/2), where N is the degree of the polynomial modulus, the number of steps to rotate must have
	/// absolute value at most N/2-1.
	///
	/// * `a` - The ciphertext to rotate
	/// * `steps` - The number of steps to rotate (positive left, negative right)
	/// * `galois_keys` - The Galois keys
	fn rotate_rows_inplace(
		&self,
		a: &Self::Ciphertext,
		steps: i32,
		galois_keys: &GaloisKey,
	) -> Result<()>;

	/// Rotates plaintext matrix columns cyclically.
	///
	/// When batching is used with the BFV scheme, this function rotates the encrypted plaintext matrix columns
	/// cyclically. Since the size of the batched matrix is 2-by-(N/2), where N is the degree of the polynomial
	/// modulus, this means simply swapping the two rows. Dynamic memory allocations in the process are allocated
	/// from the memory pool pointed to by the given MemoryPoolHandle.
	///
	/// * `encrypted` - The ciphertext to rotate
	/// * `galoisKeys` - The Galois keys
	fn rotate_columns(
		&self,
		a: &Self::Ciphertext,
		galois_keys: &GaloisKey,
	) -> Result<Self::Ciphertext>;

	/// Rotates plaintext matrix columns cyclically. This variant does so in-place.
	///
	/// When batching is used with the BFV scheme, this function rotates the encrypted plaintext matrix columns
	/// cyclically. Since the size of the batched matrix is 2-by-(N/2), where N is the degree of the polynomial
	/// modulus, this means simply swapping the two rows. Dynamic memory allocations in the process are allocated
	/// from the memory pool pointed to by the given MemoryPoolHandle.
	///
	/// * `encrypted` - The ciphertext to rotate
	/// * `galoisKeys` - The Galois keys
	fn rotate_columns_inplace(
		&self,
		a: &Self::Ciphertext,
		galois_keys: &GaloisKey,
	) -> Result<()>;

	/// Rescales a ciphertext to the next level. It helps control the noise growth in the
	/// ciphertexts.
	///
	/// * `a` - the ciphertext to rescale
	fn rescale_to_next_inplace(
		&self,
		a: &Self::Ciphertext,
	) -> Result<()>;

	/// Rescales a ciphertext to the next level. It helps control the noise growth in the
	/// ciphertexts.
	///
	/// * `a` - the ciphertext to rescale
	fn rescale_to_next(
		&self,
		a: &Self::Ciphertext,
	) -> Result<Self::Ciphertext>;

	/// Rescales a ciphertext to the next level. It helps control the noise growth in the
	/// ciphertexts.
	///
	/// * `a` - the ciphertext to rescale
	/// * parms_id - the parameters id to rescale to
	fn rescale_to(
		&self,
		a: &Self::Ciphertext,
		parms_id: &[u64],
	) -> Result<Self::Ciphertext>;

}

impl Evaluator {
	/// Creates an Evaluator instance initialized with the specified Context.
	/// * `ctx` - The context.
	pub fn new(ctx: &Context) -> Result<Self> {
		let mut handle = null_mut();

		try_seal!(unsafe { bindgen::Evaluator_Create(ctx.get_handle(), &mut handle) })?;

		Ok(Self {
			handle: AtomicPtr::new(handle),
		})
	}

	/// Gets the handle to the internal SEAL object.
	pub(crate) unsafe fn get_handle(&self) -> *mut c_void {
		self.handle.load(Ordering::SeqCst)
	}
}

impl EvaluatorOps for Evaluator {
    type Plaintext = Plaintext;
    type Ciphertext = Ciphertext;

	fn negate_inplace(
		&self,
		a: &mut Ciphertext,
	) -> Result<()> {
		try_seal!(unsafe {
			bindgen::Evaluator_Negate(self.get_handle(), a.get_handle(), a.get_handle())
		})?;

		Ok(())
	}

	fn negate(
		&self,
		a: &Ciphertext,
	) -> Result<Ciphertext> {
		let out = Ciphertext::new()?;

		try_seal!(unsafe {
			bindgen::Evaluator_Negate(self.get_handle(), a.get_handle(), out.get_handle())
		})?;

		Ok(out)
	}

	fn add_inplace(
		&self,
		a: &mut Ciphertext,
		b: &Ciphertext,
	) -> Result<()> {
		try_seal!(unsafe {
			bindgen::Evaluator_Add(
				self.get_handle(),
				a.get_handle(),
				b.get_handle(),
				a.get_handle(),
			)
		})?;

		Ok(())
	}

	fn add(
		&self,
		a: &Ciphertext,
		b: &Ciphertext,
	) -> Result<Ciphertext> {
		let c = Ciphertext::new()?;

		try_seal!(unsafe {
			bindgen::Evaluator_Add(
				self.get_handle(),
				a.get_handle(),
				b.get_handle(),
				c.get_handle(),
			)
		})?;

		Ok(c)
	}

	fn add_many(
		&self,
		a: &[Ciphertext],
	) -> Result<Ciphertext> {
		let c = Ciphertext::new()?;

		let mut a_ptr = unsafe {
			a.iter()
				.map(|x| x.get_handle())
				.collect::<Vec<*mut c_void>>()
		};

		try_seal!(unsafe {
			bindgen::Evaluator_AddMany(
				self.get_handle(),
				a_ptr.len() as u64,
				a_ptr.as_mut_ptr(),
				c.get_handle(),
			)
		})?;

		Ok(c)
	}

	fn multiply_many(
		&self,
		a: &[Ciphertext],
		relin_keys: &RelinearizationKey,
	) -> Result<Ciphertext> {
		let c = Ciphertext::new()?;

		let mut a_ptr = unsafe {
			a.iter()
				.map(|x| x.get_handle())
				.collect::<Vec<*mut c_void>>()
		};

		// let mem = MemoryPool::new()?;

		try_seal!(unsafe {
			bindgen::Evaluator_MultiplyMany(
				self.get_handle(),
				a_ptr.len() as u64,
				a_ptr.as_mut_ptr(),
				relin_keys.get_handle(),
				c.get_handle(),
				null_mut(),
				// mem.get_handle(),
			)
		})?;

		Ok(c)
	}

	fn sub_inplace(
		&self,
		a: &mut Ciphertext,
		b: &Ciphertext,
	) -> Result<()> {
		try_seal!(unsafe {
			bindgen::Evaluator_Sub(
				self.get_handle(),
				a.get_handle(),
				b.get_handle(),
				a.get_handle(),
			)
		})?;

		Ok(())
	}

	fn sub(
		&self,
		a: &Ciphertext,
		b: &Ciphertext,
	) -> Result<Ciphertext> {
		let c = Ciphertext::new()?;

		try_seal!(unsafe {
			bindgen::Evaluator_Sub(
				self.get_handle(),
				a.get_handle(),
				b.get_handle(),
				c.get_handle(),
			)
		})?;

		Ok(c)
	}

	fn multiply_inplace(
		&self,
		a: &mut Ciphertext,
		b: &Ciphertext,
	) -> Result<()> {
		try_seal!(unsafe {
			bindgen::Evaluator_Multiply(
				self.get_handle(),
				a.get_handle(),
				b.get_handle(),
				a.get_handle(),
				null_mut(),
			)
		})?;

		Ok(())
	}

	fn multiply(
		&self,
		a: &Ciphertext,
		b: &Ciphertext,
	) -> Result<Ciphertext> {
		let c = Ciphertext::new()?;

		try_seal!(unsafe {
			bindgen::Evaluator_Multiply(
				self.get_handle(),
				a.get_handle(),
				b.get_handle(),
				c.get_handle(),
				null_mut(),
			)
		})?;

		Ok(c)
	}

	fn square_inplace(
		&self,
		a: &mut Ciphertext,
	) -> Result<()> {
		try_seal!(unsafe {
			bindgen::Evaluator_Square(
				self.get_handle(),
				a.get_handle(),
				a.get_handle(),
				null_mut(),
			)
		})?;

		Ok(())
	}

	fn square(
		&self,
		a: &Ciphertext,
	) -> Result<Ciphertext> {
		let c = Ciphertext::new()?;

		try_seal!(unsafe {
			bindgen::Evaluator_Square(
				self.get_handle(),
				a.get_handle(),
				c.get_handle(),
				null_mut(),
			)
		})?;

		Ok(c)
	}

	fn mod_switch_to_next(
		&self,
		a: &Ciphertext,
	) -> Result<Ciphertext> {
		let c = Ciphertext::new()?;

		try_seal!(unsafe {
			bindgen::Evaluator_ModSwitchToNext1(
				self.get_handle(),
				a.get_handle(),
				c.get_handle(),
				null_mut(),
			)
		})?;

		Ok(c)
	}

	fn mod_switch_to_next_inplace(
		&self,
		a: &Ciphertext,
	) -> Result<()> {
		try_seal!(unsafe {
			bindgen::Evaluator_ModSwitchToNext1(
				self.get_handle(),
				a.get_handle(),
				a.get_handle(),
				null_mut(),
			)
		})?;

		Ok(())
	}

	fn mod_switch_to_next_plaintext(
		&self,
		a: &Plaintext,
	) -> Result<Plaintext> {
		let p = Plaintext::new()?;

		try_seal!(unsafe {
			bindgen::Evaluator_ModSwitchToNext2(self.get_handle(), a.get_handle(), p.get_handle())
		})?;

		Ok(p)
	}

	fn mod_switch_to_next_inplace_plaintext(
		&self,
		a: &Plaintext,
	) -> Result<()> {
		try_seal!(unsafe {
			bindgen::Evaluator_ModSwitchToNext2(self.get_handle(), a.get_handle(), a.get_handle())
		})?;

		Ok(())
	}

	fn exponentiate(
		&self,
		a: &Ciphertext,
		exponent: u64,
		relin_keys: &RelinearizationKey,
	) -> Result<Ciphertext> {
		let c = Ciphertext::new()?;

		try_seal!(unsafe {
			bindgen::Evaluator_Exponentiate(
				self.get_handle(),
				a.get_handle(),
				exponent,
				relin_keys.get_handle(),
				c.get_handle(),
				null_mut(),
			)
		})?;

		Ok(c)
	}

	fn exponentiate_inplace(
		&self,
		a: &Ciphertext,
		exponent: u64,
		relin_keys: &RelinearizationKey,
	) -> Result<()> {
		try_seal!(unsafe {
			bindgen::Evaluator_Exponentiate(
				self.get_handle(),
				a.get_handle(),
				exponent,
				relin_keys.get_handle(),
				a.get_handle(),
				null_mut(),
			)
		})?;

		Ok(())
	}

	fn add_plain(
		&self,
		a: &Ciphertext,
		b: &Plaintext,
	) -> Result<Ciphertext> {
		let c = Ciphertext::new()?;

		try_seal!(unsafe {
			bindgen::Evaluator_AddPlain(
				self.get_handle(),
				a.get_handle(),
				b.get_handle(),
				c.get_handle(),
			)
		})?;

		Ok(c)
	}

	fn add_plain_inplace(
		&self,
		a: &mut Ciphertext,
		b: &Plaintext,
	) -> Result<()> {
		try_seal!(unsafe {
			bindgen::Evaluator_AddPlain(
				self.get_handle(),
				a.get_handle(),
				b.get_handle(),
				a.get_handle(),
			)
		})?;

		Ok(())
	}

	fn sub_plain(
		&self,
		a: &Ciphertext,
		b: &Plaintext,
	) -> Result<Ciphertext> {
		let c = Ciphertext::new()?;

		try_seal!(unsafe {
			bindgen::Evaluator_SubPlain(
				self.get_handle(),
				a.get_handle(),
				b.get_handle(),
				c.get_handle(),
			)
		})?;

		Ok(c)
	}

	fn sub_plain_inplace(
		&self,
		a: &mut Ciphertext,
		b: &Plaintext,
	) -> Result<()> {
		try_seal!(unsafe {
			bindgen::Evaluator_SubPlain(
				self.get_handle(),
				a.get_handle(),
				b.get_handle(),
				a.get_handle(),
			)
		})?;

		Ok(())
	}

	fn multiply_plain(
		&self,
		a: &Ciphertext,
		b: &Plaintext,
	) -> Result<Ciphertext> {
		let c = Ciphertext::new()?;

		try_seal!(unsafe {
			bindgen::Evaluator_MultiplyPlain(
				self.get_handle(),
				a.get_handle(),
				b.get_handle(),
				c.get_handle(),
				null_mut(),
			)
		})?;

		Ok(c)
	}

	fn multiply_plain_inplace(
		&self,
		a: &mut Ciphertext,
		b: &Plaintext,
	) -> Result<()> {
		try_seal!(unsafe {
			bindgen::Evaluator_MultiplyPlain(
				self.get_handle(),
				a.get_handle(),
				b.get_handle(),
				a.get_handle(),
				null_mut(),
			)
		})?;

		Ok(())
	}

	fn relinearize_inplace(
		&self,
		a: &mut Ciphertext,
		relin_keys: &RelinearizationKey,
	) -> Result<()> {
		try_seal!(unsafe {
			bindgen::Evaluator_Relinearize(
				self.get_handle(),
				a.get_handle(),
				relin_keys.get_handle(),
				a.get_handle(),
				null_mut(),
			)
		})?;

		Ok(())
	}

	fn relinearize(
		&self,
		a: &Ciphertext,
		relin_keys: &RelinearizationKey,
	) -> Result<Ciphertext> {
		let out = Ciphertext::new()?;

		try_seal!(unsafe {
			bindgen::Evaluator_Relinearize(
				self.get_handle(),
				a.get_handle(),
				relin_keys.get_handle(),
				out.get_handle(),
				null_mut(),
			)
		})?;

		Ok(out)
	}

	fn rotate_rows(
		&self,
		a: &Ciphertext,
		steps: i32,
		galois_keys: &GaloisKey,
	) -> Result<Ciphertext> {
		let out = Ciphertext::new()?;

		try_seal!(unsafe {
			bindgen::Evaluator_RotateRows(
				self.get_handle(),
				a.get_handle(),
				steps,
				galois_keys.get_handle(),
				out.get_handle(),
				null_mut(),
			)
		})?;

		Ok(out)
	}

	fn rotate_rows_inplace(
		&self,
		a: &Ciphertext,
		steps: i32,
		galois_keys: &GaloisKey,
	) -> Result<()> {
		try_seal!(unsafe {
			bindgen::Evaluator_RotateRows(
				self.get_handle(),
				a.get_handle(),
				steps,
				galois_keys.get_handle(),
				a.get_handle(),
				null_mut(),
			)
		})?;

		Ok(())
	}

	fn rotate_columns(
		&self,
		a: &Ciphertext,
		galois_keys: &GaloisKey,
	) -> Result<Ciphertext> {
		let out = Ciphertext::new()?;

		try_seal!(unsafe {
			bindgen::Evaluator_RotateColumns(
				self.get_handle(),
				a.get_handle(),
				galois_keys.get_handle(),
				out.get_handle(),
				null_mut(),
			)
		})?;

		Ok(out)
	}

	fn rotate_columns_inplace(
		&self,
		a: &Ciphertext,
		galois_keys: &GaloisKey,
	) -> Result<()> {
		try_seal!(unsafe {
			bindgen::Evaluator_RotateColumns(
				self.get_handle(),
				a.get_handle(),
				galois_keys.get_handle(),
				a.get_handle(),
				null_mut(),
			)
		})?;

		Ok(())
	}

	fn rescale_to_next_inplace(
		&self,
		a: &Ciphertext,
	) -> Result<()> {
		try_seal!(unsafe {
			bindgen::Evaluator_RescaleToNext(
				self.get_handle(),
				a.get_handle(),
				a.get_handle(),
				null_mut(),
			)
		})?;

		Ok(())
	}

	fn rescale_to_next(
		&self,
		a: &Ciphertext,
	) -> Result<Ciphertext> {
		let c = Ciphertext::new()?;

		try_seal!(unsafe {
			bindgen::Evaluator_RescaleToNext(
				self.get_handle(),
				a.get_handle(),
				c.get_handle(),
				null_mut(),
			)
		})?;

		Ok(c)
	}

	fn rescale_to(
		&self,
		a: &Ciphertext,
		parms_id: &[u64],
	) -> Result<Ciphertext> {
		let c = Ciphertext::new()?;

		try_seal!(unsafe {
			let mut parms_id = parms_id.to_vec();
			let parms_id_ptr = parms_id.as_mut_ptr();
			bindgen::Evaluator_RescaleTo(
				self.get_handle(),
				a.get_handle(),
				parms_id_ptr,
				c.get_handle(),
				null_mut(),
			)
		})?;

		Ok(c)
	}
}

impl Drop for Evaluator {
	fn drop(&mut self) {
		try_seal!(unsafe { bindgen::Evaluator_Destroy(self.get_handle()) })
			.expect("Internal error in Evaluator::drop()");
	}
}

#[cfg(test)]
mod bfv_tests {
	use super::*;
	use crate::*;

	fn run_bfv_test<F>(test: F)
	where
		F: FnOnce(Decryptor, BFVEncoder, Encryptor<SymAsym>, Evaluator, KeyGenerator),
	{
		let params = BFVEncryptionParametersBuilder::new()
			.set_poly_modulus_degree(DegreeType::D8192)
			.set_coefficient_modulus(
				CoefficientModulusFactory::build(DegreeType::D8192, &[50, 30, 30, 50, 50]).unwrap(),
			)
			.set_plain_modulus(PlainModulusFactory::batching(DegreeType::D8192, 32).unwrap())
			.build()
			.unwrap();

		let ctx = Context::new(&params, false, SecurityLevel::TC128).unwrap();
		let gen = KeyGenerator::new(&ctx).unwrap();

		let encoder = BFVEncoder::new(&ctx).unwrap();

		let public_key = gen.create_public_key();
		let secret_key = gen.secret_key();

		let encryptor =
			Encryptor::with_public_and_secret_key(&ctx, &public_key, &secret_key).unwrap();
		let decryptor = Decryptor::new(&ctx, &secret_key).unwrap();
		let evaluator = Evaluator::new(&ctx).unwrap();

		test(decryptor, encoder, encryptor, evaluator, gen);
	}

	fn make_vec(encoder: &BFVEncoder) -> Vec<i64> {
		let mut data = vec![];

		for i in 0..encoder.get_slot_count() {
			data.push(encoder.get_slot_count() as i64 / 2i64 - i as i64)
		}

		data
	}

	fn make_small_vec(encoder: &BFVEncoder) -> Vec<i64> {
		let mut data = vec![];

		for i in 0..encoder.get_slot_count() {
			data.push(16i64 - i as i64 % 32i64);
		}

		data
	}

	#[test]
	fn can_create_and_destroy_evaluator() {
		let params = BFVEncryptionParametersBuilder::new()
			.set_poly_modulus_degree(DegreeType::D8192)
			.set_coefficient_modulus(
				CoefficientModulusFactory::build(DegreeType::D8192, &[50, 30, 30, 50, 50]).unwrap(),
			)
			.set_plain_modulus(PlainModulusFactory::batching(DegreeType::D8192, 20).unwrap())
			.build()
			.unwrap();

		let ctx = Context::new(&params, false, SecurityLevel::TC128).unwrap();

		let evaluator = EvaluatorBase::new(&ctx);

		std::mem::drop(evaluator);
	}

	#[test]
	fn can_negate() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();

			let b_c = evaluator.negate(&a_c).unwrap();

			let b_p = decryptor.decrypt(&b_c).unwrap();
			let b: Vec<i64> = encoder.decode_i64(&b_p).unwrap();

			assert_eq!(a.len(), b.len());

			for i in 0..a.len() {
				assert_eq!(a[i], -b[i]);
			}
		});
	}

	#[test]
	fn can_negate_inplace() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let mut a_c = encryptor.encrypt(&a_p).unwrap();

			evaluator.negate_inplace(&mut a_c).unwrap();

			let a_p = decryptor.decrypt(&a_c).unwrap();
			let b: Vec<i64> = encoder.decode_i64(&a_p).unwrap();

			assert_eq!(a.len(), b.len());

			for i in 0..a.len() {
				assert_eq!(a[i], -b[i]);
			}
		});
	}

	#[test]
	fn can_add() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let b_p = encoder.encode_i64(&b).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();
			let b_c = encryptor.encrypt(&b_p).unwrap();

			let c_c = evaluator.add(&a_c, &b_c).unwrap();

			let c_p = decryptor.decrypt(&c_c).unwrap();
			let c: Vec<i64> = encoder.decode_i64(&c_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				assert_eq!(c[i], a[i] + b[i]);
			}
		});
	}

	#[test]
	fn can_add_inplace() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let b_p = encoder.encode_i64(&b).unwrap();
			let mut a_c = encryptor.encrypt(&a_p).unwrap();
			let b_c = encryptor.encrypt(&b_p).unwrap();

			evaluator.add_inplace(&mut a_c, &b_c).unwrap();

			let a_p = decryptor.decrypt(&a_c).unwrap();
			let c: Vec<i64> = encoder.decode_i64(&a_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				assert_eq!(c[i], a[i] + b[i]);
			}
		});
	}

	#[test]
	fn can_add_many() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let c = make_vec(&encoder);
			let d = make_vec(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let b_p = encoder.encode_i64(&b).unwrap();
			let c_p = encoder.encode_i64(&c).unwrap();
			let d_p = encoder.encode_i64(&d).unwrap();

			let data_c = vec![
				encryptor.encrypt(&a_p).unwrap(),
				encryptor.encrypt(&b_p).unwrap(),
				encryptor.encrypt(&c_p).unwrap(),
				encryptor.encrypt(&d_p).unwrap(),
			];

			let out_c = evaluator.add_many(&data_c).unwrap();

			let out_p = decryptor.decrypt(&out_c).unwrap();
			let out: Vec<i64> = encoder.decode_i64(&out_p).unwrap();

			assert_eq!(a.len(), out.len());
			assert_eq!(b.len(), out.len());
			assert_eq!(c.len(), out.len());
			assert_eq!(d.len(), out.len());

			for i in 0..a.len() {
				assert_eq!(out[i], a[i] + b[i] + c[i] + d[i]);
			}
		});
	}

	#[test]
	fn can_multiply_many() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, keygen| {
			let relin_keys = keygen.create_relinearization_keys().unwrap();

			let a = make_small_vec(&encoder);
			let b = make_small_vec(&encoder);
			let c = make_small_vec(&encoder);
			let d = make_small_vec(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let b_p = encoder.encode_i64(&b).unwrap();
			let c_p = encoder.encode_i64(&c).unwrap();
			let d_p = encoder.encode_i64(&d).unwrap();

			let data_c = vec![
				encryptor.encrypt(&a_p).unwrap(),
				encryptor.encrypt(&b_p).unwrap(),
				encryptor.encrypt(&c_p).unwrap(),
				encryptor.encrypt(&d_p).unwrap(),
			];

			let out_c = evaluator.multiply_many(&data_c, &relin_keys).unwrap();

			let out_p = decryptor.decrypt(&out_c).unwrap();
			let out: Vec<i64> = encoder.decode_i64(&out_p).unwrap();

			assert_eq!(a.len(), out.len());
			assert_eq!(b.len(), out.len());
			assert_eq!(c.len(), out.len());
			assert_eq!(d.len(), out.len());

			for i in 0..a.len() {
				assert_eq!(out[i], a[i] * b[i] * c[i] * d[i]);
			}
		});
	}

	#[test]
	fn can_sub() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let b_p = encoder.encode_i64(&b).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();
			let b_c = encryptor.encrypt(&b_p).unwrap();

			let c_c = evaluator.sub(&a_c, &b_c).unwrap();

			let c_p = decryptor.decrypt(&c_c).unwrap();
			let c: Vec<i64> = encoder.decode_i64(&c_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				assert_eq!(c[i], a[i] - b[i]);
			}
		});
	}

	#[test]
	fn can_sub_inplace() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let b_p = encoder.encode_i64(&b).unwrap();
			let mut a_c = encryptor.encrypt(&a_p).unwrap();
			let b_c = encryptor.encrypt(&b_p).unwrap();

			evaluator.sub_inplace(&mut a_c, &b_c).unwrap();

			let a_p = decryptor.decrypt(&a_c).unwrap();
			let c: Vec<i64> = encoder.decode_i64(&a_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				assert_eq!(c[i], a[i] - b[i]);
			}
		});
	}

	#[test]
	fn can_multiply() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let b_p = encoder.encode_i64(&b).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();
			let b_c = encryptor.encrypt(&b_p).unwrap();

			let c_c = evaluator.multiply(&a_c, &b_c).unwrap();

			let c_p = decryptor.decrypt(&c_c).unwrap();
			let c: Vec<i64> = encoder.decode_i64(&c_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				assert_eq!(c[i], a[i] * b[i]);
			}
		});
	}

	#[test]
	fn can_multiply_inplace() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let b_p = encoder.encode_i64(&b).unwrap();
			let mut a_c = encryptor.encrypt(&a_p).unwrap();
			let b_c = encryptor.encrypt(&b_p).unwrap();

			evaluator.multiply_inplace(&mut a_c, &b_c).unwrap();

			let a_p = decryptor.decrypt(&a_c).unwrap();
			let c: Vec<i64> = encoder.decode_i64(&a_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				assert_eq!(c[i], a[i] * b[i]);
			}
		});
	}

	#[test]
	fn can_square() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();

			let b_c = evaluator.square(&a_c).unwrap();

			let b_p = decryptor.decrypt(&b_c).unwrap();
			let b: Vec<i64> = encoder.decode_i64(&b_p).unwrap();

			assert_eq!(a.len(), b.len());

			for i in 0..a.len() {
				assert_eq!(b[i], a[i] * a[i]);
			}
		});
	}

	#[test]
	fn can_square_inplace() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let mut a_c = encryptor.encrypt(&a_p).unwrap();

			evaluator.square_inplace(&mut a_c).unwrap();

			let a_p = decryptor.decrypt(&a_c).unwrap();
			let b: Vec<i64> = encoder.decode_i64(&a_p).unwrap();

			assert_eq!(a.len(), b.len());

			for i in 0..a.len() {
				assert_eq!(b[i], a[i] * a[i]);
			}
		});
	}

	#[test]
	fn can_relinearize_inplace() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, keygen| {
			let relin_keys = keygen.create_relinearization_keys().unwrap();

			let a = make_vec(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let mut a_c = encryptor.encrypt(&a_p).unwrap();
			let mut a_c_2 = encryptor.encrypt(&a_p).unwrap();

			let noise_before = decryptor.invariant_noise_budget(&a_c).unwrap();

			evaluator.square_inplace(&mut a_c).unwrap();
			evaluator
				.relinearize_inplace(&mut a_c, &relin_keys)
				.unwrap();
			evaluator.square_inplace(&mut a_c).unwrap();
			evaluator
				.relinearize_inplace(&mut a_c, &relin_keys)
				.unwrap();

			let relin_noise = noise_before - decryptor.invariant_noise_budget(&a_c).unwrap();

			let noise_before = decryptor.invariant_noise_budget(&a_c_2).unwrap();

			evaluator.square_inplace(&mut a_c_2).unwrap();
			evaluator.square_inplace(&mut a_c_2).unwrap();

			let no_relin_noise = noise_before - decryptor.invariant_noise_budget(&a_c_2).unwrap();

			assert!(relin_noise < no_relin_noise)
		});
	}

	#[test]
	fn can_relinearize() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, keygen| {
			let relin_keys = keygen.create_relinearization_keys().unwrap();

			let a = make_vec(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let mut a_c = encryptor.encrypt(&a_p).unwrap();
			let mut a_c_2 = encryptor.encrypt(&a_p).unwrap();

			let noise_before = decryptor.invariant_noise_budget(&a_c).unwrap();

			evaluator.square_inplace(&mut a_c).unwrap();
			let mut a_c = evaluator.relinearize(&a_c, &relin_keys).unwrap();
			evaluator.square_inplace(&mut a_c).unwrap();
			let a_c = evaluator.relinearize(&a_c, &relin_keys).unwrap();

			let relin_noise = noise_before - decryptor.invariant_noise_budget(&a_c).unwrap();

			let noise_before = decryptor.invariant_noise_budget(&a_c_2).unwrap();

			evaluator.square_inplace(&mut a_c_2).unwrap();
			evaluator.square_inplace(&mut a_c_2).unwrap();

			let no_relin_noise = noise_before - decryptor.invariant_noise_budget(&a_c_2).unwrap();

			assert!(relin_noise < no_relin_noise)
		});
	}

	#[test]
	fn can_exponentiate() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, keygen| {
			let relin_keys = keygen.create_relinearization_keys().unwrap();

			let a = make_small_vec(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();

			let c_c = evaluator.exponentiate(&a_c, 4, &relin_keys).unwrap();

			let c_p = decryptor.decrypt(&c_c).unwrap();
			let c: Vec<i64> = encoder.decode_i64(&c_p).unwrap();

			assert_eq!(a.len(), c.len());

			for i in 0..a.len() {
				assert_eq!(c[i], a[i] * a[i] * a[i] * a[i]);
			}
		});
	}

	#[test]
	fn can_exponentiate_inplace() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, keygen| {
			let relin_keys = keygen.create_relinearization_keys().unwrap();

			let a = make_small_vec(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();

			evaluator
				.exponentiate_inplace(&a_c, 4, &relin_keys)
				.unwrap();

			let a_p = decryptor.decrypt(&a_c).unwrap();
			let c: Vec<i64> = encoder.decode_i64(&a_p).unwrap();

			assert_eq!(a.len(), c.len());

			for i in 0..a.len() {
				assert_eq!(c[i], a[i] * a[i] * a[i] * a[i]);
			}
		});
	}

	#[test]
	fn can_add_plain() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let b_p = encoder.encode_i64(&b).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();

			let c_c = evaluator.add_plain(&a_c, &b_p).unwrap();

			let c_p = decryptor.decrypt(&c_c).unwrap();
			let c: Vec<i64> = encoder.decode_i64(&c_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				assert_eq!(c[i], a[i] + b[i]);
			}
		});
	}

	#[test]
	fn can_add_plain_inplace() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let b_p = encoder.encode_i64(&b).unwrap();
			let mut a_c = encryptor.encrypt(&a_p).unwrap();

			evaluator.add_plain_inplace(&mut a_c, &b_p).unwrap();

			let a_p = decryptor.decrypt(&a_c).unwrap();
			let c: Vec<i64> = encoder.decode_i64(&a_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				assert_eq!(c[i], a[i] + b[i]);
			}
		});
	}

	#[test]
	fn can_sub_plain() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let b_p = encoder.encode_i64(&b).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();

			let c_c = evaluator.sub_plain(&a_c, &b_p).unwrap();

			let c_p = decryptor.decrypt(&c_c).unwrap();
			let c: Vec<i64> = encoder.decode_i64(&c_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				assert_eq!(c[i], a[i] - b[i]);
			}
		});
	}

	#[test]
	fn can_sub_plain_inplace() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let b_p = encoder.encode_i64(&b).unwrap();
			let mut a_c = encryptor.encrypt(&a_p).unwrap();

			evaluator.sub_plain_inplace(&mut a_c, &b_p).unwrap();

			let a_p = decryptor.decrypt(&a_c).unwrap();
			let c: Vec<i64> = encoder.decode_i64(&a_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				assert_eq!(c[i], a[i] - b[i]);
			}
		});
	}

	#[test]
	fn can_multiply_plain() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let b_p = encoder.encode_i64(&b).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();

			let c_c = evaluator.multiply_plain(&a_c, &b_p).unwrap();

			let c_p = decryptor.decrypt(&c_c).unwrap();
			let c: Vec<i64> = encoder.decode_i64(&c_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				assert_eq!(c[i], a[i] * b[i]);
			}
		});
	}

	#[test]
	fn can_multiply_plain_inplace() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let b_p = encoder.encode_i64(&b).unwrap();
			let mut a_c = encryptor.encrypt(&a_p).unwrap();

			evaluator.multiply_plain_inplace(&mut a_c, &b_p).unwrap();

			let a_p = decryptor.decrypt(&a_c).unwrap();
			let c: Vec<i64> = encoder.decode_i64(&a_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				assert_eq!(c[i], a[i] * b[i]);
			}
		});
	}

	fn make_matrix(encoder: &BFVEncoder) -> Vec<i64> {
		let dim = encoder.get_slot_count();
		let dim_2 = dim / 2;

		let mut matrix = vec![0i64; dim];

		matrix[0] = 1;
		matrix[1] = -2;
		matrix[dim_2] = -1;
		matrix[dim_2 + 1] = 2;

		matrix
	}

	#[test]
	fn can_rotate_rows() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, keygen| {
			let galois_keys = keygen.create_galois_keys();

			let a = make_matrix(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();

			let c_c = evaluator
				.rotate_rows(&a_c, -1, &galois_keys.unwrap())
				.unwrap();

			let c_p = decryptor.decrypt(&c_c).unwrap();
			let c: Vec<i64> = encoder.decode_i64(&c_p).unwrap();

			assert_eq!(a[0], c[1]);
			assert_eq!(a[1], c[2]);
			assert_eq!(a[4096], c[4097]);
			assert_eq!(a[4097], c[4098]);
		});
	}

	#[test]
	fn can_rotate_rows_inplace() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, keygen| {
			let galois_keys = keygen.create_galois_keys();

			let a = make_matrix(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();

			evaluator
				.rotate_rows_inplace(&a_c, -1, &galois_keys.unwrap())
				.unwrap();

			let a_p = decryptor.decrypt(&a_c).unwrap();
			let c: Vec<i64> = encoder.decode_i64(&a_p).unwrap();

			assert_eq!(a[0], c[1]);
			assert_eq!(a[1], c[2]);
			assert_eq!(a[4096], c[4097]);
			assert_eq!(a[4097], c[4098]);
		});
	}

	#[test]
	fn can_rotate_columns() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, keygen| {
			let galois_keys = keygen.create_galois_keys();

			let a = make_matrix(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();

			let c_c = evaluator
				.rotate_columns(&a_c, &galois_keys.unwrap())
				.unwrap();

			let c_p = decryptor.decrypt(&c_c).unwrap();
			let c: Vec<i64> = encoder.decode_i64(&c_p).unwrap();

			assert_eq!(a[0], c[4096]);
			assert_eq!(a[1], c[4097]);
			assert_eq!(a[4096], c[0]);
			assert_eq!(a[4097], c[1]);
		});
	}

	#[test]
	fn can_rotate_columns_inplace() {
		run_bfv_test(|decryptor, encoder, encryptor, evaluator, keygen| {
			let galois_keys = keygen.create_galois_keys();

			let a = make_matrix(&encoder);
			let a_p = encoder.encode_i64(&a).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();

			evaluator
				.rotate_columns_inplace(&a_c, &galois_keys.unwrap())
				.unwrap();

			let a_p = decryptor.decrypt(&a_c).unwrap();
			let c: Vec<i64> = encoder.decode_i64(&a_p).unwrap();

			assert_eq!(a[0], c[4096]);
			assert_eq!(a[1], c[4097]);
			assert_eq!(a[4096], c[0]);
			assert_eq!(a[4097], c[1]);
		});
	}
}


#[cfg(test)]
mod ckks_tests {
	use super::*;
	use crate::*;

	fn float_assert_eq(
		a: f64,
		b: f64,
	) {
		assert!((a - b).abs() < 0.0001);
	}

	fn run_ckks_test<F>(test: F)
	where
		F: FnOnce(Decryptor, CKKSEncoder, Encryptor<SymAsym>, Evaluator, KeyGenerator),
	{
		let params = CKKSEncryptionParametersBuilder::new()
			.set_poly_modulus_degree(DegreeType::D8192)
			.set_coefficient_modulus(
				CoefficientModulusFactory::build(DegreeType::D8192, &[60, 40, 40, 60]).unwrap(),
			)
			.build()
			.unwrap();

		let ctx = Context::new(&params, false, SecurityLevel::TC128).unwrap();
		let gen = KeyGenerator::new(&ctx).unwrap();

		let scale = 2.0f64.powi(40);
		let encoder = CKKSEncoder::new(&ctx, scale).unwrap();

		let public_key = gen.create_public_key();
		let secret_key = gen.secret_key();

		let encryptor =
			Encryptor::with_public_and_secret_key(&ctx, &public_key, &secret_key).unwrap();
		let decryptor = Decryptor::new(&ctx, &secret_key).unwrap();
		let evaluator = Evaluator::new(&ctx).unwrap();

		test(decryptor, encoder, encryptor, evaluator, gen);
	}

	fn make_vec(encoder: &CKKSEncoder) -> Vec<f64> {
		let mut data = vec![];

		for i in 0..encoder.get_slot_count() {
			data.push(encoder.get_slot_count() as f64 / 2f64 - i as f64)
		}

		data
	}

	fn make_small_vec(encoder: &CKKSEncoder) -> Vec<f64> {
		let mut data = vec![];

		for i in 0..encoder.get_slot_count() {
			data.push(16f64 - i as f64 % 32f64);
		}

		data
	}

	#[test]
	fn can_create_and_destroy_evaluator() {
		let params = CKKSEncryptionParametersBuilder::new()
			.set_poly_modulus_degree(DegreeType::D8192)
			.set_coefficient_modulus(
				CoefficientModulusFactory::build(DegreeType::D8192, &[60, 40, 40, 60]).unwrap(),
			)
			.build()
			.unwrap();

		let ctx = Context::new(&params, false, SecurityLevel::TC128).unwrap();

		let evaluator = EvaluatorBase::new(&ctx);

		std::mem::drop(evaluator);
	}

	#[test]
	fn can_negate() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();

			let b_c = evaluator.negate(&a_c).unwrap();

			let b_p = decryptor.decrypt(&b_c).unwrap();
			let b = encoder.decode_f64(&b_p).unwrap();

			assert_eq!(a.len(), b.len());

			for i in 0..a.len() {
				float_assert_eq(a[i], -b[i]);
			}
		});
	}

	#[test]
	fn can_negate_inplace() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let mut a_c = encryptor.encrypt(&a_p).unwrap();

			evaluator.negate_inplace(&mut a_c).unwrap();

			let a_p = decryptor.decrypt(&a_c).unwrap();
			let b = encoder.decode_f64(&a_p).unwrap();

			assert_eq!(a.len(), b.len());

			for i in 0..a.len() {
				float_assert_eq(a[i], -b[i]);
			}
		});
	}

	#[test]
	fn can_add() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let b_p = encoder.encode_f64(&b).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();
			let b_c = encryptor.encrypt(&b_p).unwrap();

			let c_c = evaluator.add(&a_c, &b_c).unwrap();

			let c_p = decryptor.decrypt(&c_c).unwrap();
			let c = encoder.decode_f64(&c_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				float_assert_eq(c[i], a[i] + b[i]);
			}
		});
	}

	#[test]
	fn can_add_inplace() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let b_p = encoder.encode_f64(&b).unwrap();
			let mut a_c = encryptor.encrypt(&a_p).unwrap();
			let b_c = encryptor.encrypt(&b_p).unwrap();

			evaluator.add_inplace(&mut a_c, &b_c).unwrap();

			let a_p = decryptor.decrypt(&a_c).unwrap();
			let c = encoder.decode_f64(&a_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				float_assert_eq(c[i], a[i] + b[i]);
			}
		});
	}

	#[test]
	fn can_add_many() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let c = make_vec(&encoder);
			let d = make_vec(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let b_p = encoder.encode_f64(&b).unwrap();
			let c_p = encoder.encode_f64(&c).unwrap();
			let d_p = encoder.encode_f64(&d).unwrap();

			let data_c = vec![
				encryptor.encrypt(&a_p).unwrap(),
				encryptor.encrypt(&b_p).unwrap(),
				encryptor.encrypt(&c_p).unwrap(),
				encryptor.encrypt(&d_p).unwrap(),
			];

			let out_c = evaluator.add_many(&data_c).unwrap();

			let out_p = decryptor.decrypt(&out_c).unwrap();
			let out = encoder.decode_f64(&out_p).unwrap();

			assert_eq!(a.len(), out.len());
			assert_eq!(b.len(), out.len());
			assert_eq!(c.len(), out.len());
			assert_eq!(d.len(), out.len());

			for i in 0..a.len() {
				float_assert_eq(out[i], a[i] + b[i] + c[i] + d[i]);
			}
		});
	}

	#[test]
	#[ignore = "CKKS multiply many is not yet working"]
	fn can_multiply_many() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, keygen| {
			let relin_keys = keygen.create_relinearization_keys().unwrap();

			let a = make_small_vec(&encoder);
			let b = make_small_vec(&encoder);
			let c = make_small_vec(&encoder);
			let d = make_small_vec(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let b_p = encoder.encode_f64(&b).unwrap();
			let c_p = encoder.encode_f64(&c).unwrap();
			let d_p = encoder.encode_f64(&d).unwrap();

			let data_c = vec![
				encryptor.encrypt(&a_p).unwrap(),
				encryptor.encrypt(&b_p).unwrap(),
				encryptor.encrypt(&c_p).unwrap(),
				encryptor.encrypt(&d_p).unwrap(),
			];

			let out_c = evaluator.multiply_many(&data_c, &relin_keys).unwrap();

			let out_p = decryptor.decrypt(&out_c).unwrap();
			let out = encoder.decode_f64(&out_p).unwrap();

			assert_eq!(a.len(), out.len());
			assert_eq!(b.len(), out.len());
			assert_eq!(c.len(), out.len());
			assert_eq!(d.len(), out.len());

			for i in 0..a.len() {
				float_assert_eq(out[i], a[i] * b[i] * c[i] * d[i]);
			}
		});
	}

	#[test]
	fn can_sub() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let b_p = encoder.encode_f64(&b).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();
			let b_c = encryptor.encrypt(&b_p).unwrap();

			let c_c = evaluator.sub(&a_c, &b_c).unwrap();

			let c_p = decryptor.decrypt(&c_c).unwrap();
			let c = encoder.decode_f64(&c_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				float_assert_eq(c[i], a[i] - b[i]);
			}
		});
	}

	#[test]
	fn can_sub_inplace() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let b_p = encoder.encode_f64(&b).unwrap();
			let mut a_c = encryptor.encrypt(&a_p).unwrap();
			let b_c = encryptor.encrypt(&b_p).unwrap();

			evaluator.sub_inplace(&mut a_c, &b_c).unwrap();

			let a_p = decryptor.decrypt(&a_c).unwrap();
			let c = encoder.decode_f64(&a_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				float_assert_eq(c[i], a[i] - b[i]);
			}
		});
	}

	#[test]
	fn can_multiply() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let b_p = encoder.encode_f64(&b).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();
			let b_c = encryptor.encrypt(&b_p).unwrap();

			let c_c = evaluator.multiply(&a_c, &b_c).unwrap();

			let c_p = decryptor.decrypt(&c_c).unwrap();
			let c = encoder.decode_f64(&c_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				float_assert_eq(c[i], a[i] * b[i]);
			}
		});
	}

	#[test]
	fn can_multiply_inplace() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let b_p = encoder.encode_f64(&b).unwrap();
			let mut a_c = encryptor.encrypt(&a_p).unwrap();
			let b_c = encryptor.encrypt(&b_p).unwrap();

			evaluator.multiply_inplace(&mut a_c, &b_c).unwrap();

			let a_p = decryptor.decrypt(&a_c).unwrap();
			let c = encoder.decode_f64(&a_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				float_assert_eq(c[i], a[i] * b[i]);
			}
		});
	}

	#[test]
	fn can_square() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();

			let b_c = evaluator.square(&a_c).unwrap();

			let b_p = decryptor.decrypt(&b_c).unwrap();
			let b = encoder.decode_f64(&b_p).unwrap();

			assert_eq!(a.len(), b.len());

			for i in 0..a.len() {
				float_assert_eq(b[i], a[i] * a[i]);
			}
		});
	}

	#[test]
	fn can_square_inplace() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let mut a_c = encryptor.encrypt(&a_p).unwrap();

			evaluator.square_inplace(&mut a_c).unwrap();

			let a_p = decryptor.decrypt(&a_c).unwrap();
			let b = encoder.decode_f64(&a_p).unwrap();

			assert_eq!(a.len(), b.len());

			for i in 0..a.len() {
				float_assert_eq(b[i], a[i] * a[i]);
			}
		});
	}

	#[test]
	#[ignore = "CKKS relinearize is not yet working"]
	fn can_relinearize_inplace() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, keygen| {
			let relin_keys = keygen.create_relinearization_keys().unwrap();

			let a = make_vec(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let mut a_c = encryptor.encrypt(&a_p).unwrap();
			let mut a_c_2 = encryptor.encrypt(&a_p).unwrap();

			let noise_before = decryptor.invariant_noise_budget(&a_c).unwrap();

			evaluator.square_inplace(&mut a_c).unwrap();
			evaluator
				.relinearize_inplace(&mut a_c, &relin_keys)
				.unwrap();
			evaluator.square_inplace(&mut a_c).unwrap();
			evaluator
				.relinearize_inplace(&mut a_c, &relin_keys)
				.unwrap();

			let relin_noise = noise_before - decryptor.invariant_noise_budget(&a_c).unwrap();

			let noise_before = decryptor.invariant_noise_budget(&a_c_2).unwrap();

			evaluator.square_inplace(&mut a_c_2).unwrap();
			evaluator.square_inplace(&mut a_c_2).unwrap();

			let no_relin_noise = noise_before - decryptor.invariant_noise_budget(&a_c_2).unwrap();

			assert!(relin_noise < no_relin_noise)
		});
	}

	#[test]
	#[ignore = "CKKS relinearize is not yet working"]
	fn can_relinearize() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, keygen| {
			let relin_keys = keygen.create_relinearization_keys().unwrap();

			let a = make_vec(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let mut a_c = encryptor.encrypt(&a_p).unwrap();
			let mut a_c_2 = encryptor.encrypt(&a_p).unwrap();

			let noise_before = decryptor.invariant_noise_budget(&a_c).unwrap();

			evaluator.square_inplace(&mut a_c).unwrap();
			let mut a_c = evaluator.relinearize(&a_c, &relin_keys).unwrap();
			evaluator.square_inplace(&mut a_c).unwrap();
			let a_c = evaluator.relinearize(&a_c, &relin_keys).unwrap();

			let relin_noise = noise_before - decryptor.invariant_noise_budget(&a_c).unwrap();

			let noise_before = decryptor.invariant_noise_budget(&a_c_2).unwrap();

			evaluator.square_inplace(&mut a_c_2).unwrap();
			evaluator.square_inplace(&mut a_c_2).unwrap();

			let no_relin_noise = noise_before - decryptor.invariant_noise_budget(&a_c_2).unwrap();

			assert!(relin_noise < no_relin_noise)
		});
	}

	#[test]
	#[ignore = "CKKS exponentiation is not yet working"]
	fn can_exponentiate() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, keygen| {
			let relin_keys = keygen.create_relinearization_keys().unwrap();

			let a = make_small_vec(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();

			let c_c = evaluator.exponentiate(&a_c, 4, &relin_keys).unwrap();

			let c_p = decryptor.decrypt(&c_c).unwrap();
			let c = encoder.decode_f64(&c_p).unwrap();

			assert_eq!(a.len(), c.len());

			for i in 0..a.len() {
				float_assert_eq(c[i], a[i] * a[i] * a[i] * a[i]);
			}
		});
	}

	#[test]
	#[ignore = "CKKS exponentiation is not yet working"]
	fn can_exponentiate_inplace() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, keygen| {
			let relin_keys = keygen.create_relinearization_keys().unwrap();

			let a = make_small_vec(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();

			evaluator
				.exponentiate_inplace(&a_c, 4, &relin_keys)
				.unwrap();

			let a_p = decryptor.decrypt(&a_c).unwrap();
			let c = encoder.decode_f64(&a_p).unwrap();

			assert_eq!(a.len(), c.len());

			for i in 0..a.len() {
				float_assert_eq(c[i], a[i] * a[i] * a[i] * a[i]);
			}
		});
	}

	#[test]
	fn can_add_plain() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let b_p = encoder.encode_f64(&b).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();

			let c_c = evaluator.add_plain(&a_c, &b_p).unwrap();

			let c_p = decryptor.decrypt(&c_c).unwrap();
			let c = encoder.decode_f64(&c_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				float_assert_eq(c[i], a[i] + b[i]);
			}
		});
	}

	#[test]
	fn can_add_plain_inplace() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let b_p = encoder.encode_f64(&b).unwrap();
			let mut a_c = encryptor.encrypt(&a_p).unwrap();

			evaluator.add_plain_inplace(&mut a_c, &b_p).unwrap();

			let a_p = decryptor.decrypt(&a_c).unwrap();
			let c = encoder.decode_f64(&a_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				float_assert_eq(c[i], a[i] + b[i]);
			}
		});
	}

	#[test]
	fn can_sub_plain() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let b_p = encoder.encode_f64(&b).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();

			let c_c = evaluator.sub_plain(&a_c, &b_p).unwrap();

			let c_p = decryptor.decrypt(&c_c).unwrap();
			let c = encoder.decode_f64(&c_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				float_assert_eq(c[i], a[i] - b[i]);
			}
		});
	}

	#[test]
	fn can_sub_plain_inplace() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let b_p = encoder.encode_f64(&b).unwrap();
			let mut a_c = encryptor.encrypt(&a_p).unwrap();

			evaluator.sub_plain_inplace(&mut a_c, &b_p).unwrap();

			let a_p = decryptor.decrypt(&a_c).unwrap();
			let c = encoder.decode_f64(&a_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				float_assert_eq(c[i], a[i] - b[i]);
			}
		});
	}

	#[test]
	fn can_multiply_plain() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let b_p = encoder.encode_f64(&b).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();

			let c_c = evaluator.multiply_plain(&a_c, &b_p).unwrap();

			let c_p = decryptor.decrypt(&c_c).unwrap();
			let c = encoder.decode_f64(&c_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				float_assert_eq(c[i], a[i] * b[i]);
			}
		});
	}

	#[test]
	fn can_multiply_plain_inplace() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, _| {
			let a = make_vec(&encoder);
			let b = make_vec(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let b_p = encoder.encode_f64(&b).unwrap();
			let mut a_c = encryptor.encrypt(&a_p).unwrap();

			evaluator.multiply_plain_inplace(&mut a_c, &b_p).unwrap();

			let a_p = decryptor.decrypt(&a_c).unwrap();
			let c = encoder.decode_f64(&a_p).unwrap();

			assert_eq!(a.len(), c.len());
			assert_eq!(b.len(), c.len());

			for i in 0..a.len() {
				float_assert_eq(c[i], a[i] * b[i]);
			}
		});
	}

	fn make_matrix(encoder: &CKKSEncoder) -> Vec<f64> {
		let dim = encoder.get_slot_count();
		let dim_2 = dim / 2;

		let mut matrix = vec![0f64; dim];

		matrix[0] = 1f64;
		matrix[1] = -2f64;
		matrix[dim_2] = -1f64;
		matrix[dim_2 + 1] = 2f64;

		matrix
	}

	#[test]
	#[ignore = "CKKS rotate rows is not yet working"]
	fn can_rotate_rows() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, keygen| {
			let galois_keys = keygen.create_galois_keys();

			let a = make_matrix(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();

			let c_c = evaluator
				.rotate_rows(&a_c, -1, &galois_keys.unwrap())
				.unwrap();

			let c_p = decryptor.decrypt(&c_c).unwrap();
			let c = encoder.decode_f64(&c_p).unwrap();

			float_assert_eq(a[0], c[1]);
			float_assert_eq(a[1], c[2]);
			float_assert_eq(a[4096], c[4097]);
			float_assert_eq(a[4097], c[4098]);
		});
	}

	#[test]
	#[ignore = "CKKS rotate rows is not yet working"]
	fn can_rotate_rows_inplace() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, keygen| {
			let galois_keys = keygen.create_galois_keys();

			let a = make_matrix(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();

			evaluator
				.rotate_rows_inplace(&a_c, -1, &galois_keys.unwrap())
				.unwrap();

			let a_p = decryptor.decrypt(&a_c).unwrap();
			let c = encoder.decode_f64(&a_p).unwrap();

			float_assert_eq(a[0], c[1]);
			float_assert_eq(a[1], c[2]);
			float_assert_eq(a[4096], c[4097]);
			float_assert_eq(a[4097], c[4098]);
		});
	}

	#[test]
	#[ignore = "CKKS rotate columns is not yet working"]
	fn can_rotate_columns() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, keygen| {
			let galois_keys = keygen.create_galois_keys();

			let a = make_matrix(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();

			let c_c = evaluator
				.rotate_columns(&a_c, &galois_keys.unwrap())
				.unwrap();

			let c_p = decryptor.decrypt(&c_c).unwrap();
			let c = encoder.decode_f64(&c_p).unwrap();

			float_assert_eq(a[0], c[4096]);
			float_assert_eq(a[1], c[4097]);
			float_assert_eq(a[4096], c[0]);
			float_assert_eq(a[4097], c[1]);
		});
	}

	#[test]
	#[ignore = "CKKS rotate columns is not yet working"]
	fn can_rotate_columns_inplace() {
		run_ckks_test(|decryptor, encoder, encryptor, evaluator, keygen| {
			let galois_keys = keygen.create_galois_keys();

			let a = make_matrix(&encoder);
			let a_p = encoder.encode_f64(&a).unwrap();
			let a_c = encryptor.encrypt(&a_p).unwrap();

			evaluator
				.rotate_columns_inplace(&a_c, &galois_keys.unwrap())
				.unwrap();

			let a_p = decryptor.decrypt(&a_c).unwrap();
			let c = encoder.decode_f64(&a_p).unwrap();

			float_assert_eq(a[0], c[4096]);
			float_assert_eq(a[1], c[4097]);
			float_assert_eq(a[4096], c[0]);
			float_assert_eq(a[4097], c[1]);
		});
	}
}
