use pyo3::prelude::*;

use crate::{
	ciphertext::PyCiphertext, context::PyContext, keys::PyGaloisKey, keys::PyRelinearizationKey,
	plaintext::PyPlaintext,
};
use sealy::EvaluatorOps;

/// An evaluator that contains additional operations specific to the BFV scheme.
#[pyclass(module = "sealy", name = "Evaluator")]
pub struct PyEvaluator {
	pub(crate) inner: sealy::Evaluator,
}

#[pymethods]
impl PyEvaluator {
	/// Creates a Evaluator instance initialized with the specified Context.
	///  * `ctx` - The context.
	#[new]
	pub fn new(ctx: &PyContext) -> PyResult<Self> {
		let evaluator = sealy::Evaluator::new(&ctx.inner).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to create Evaluator: {:?}",
				e
			))
		})?;
		Ok(Self {
			inner: evaluator,
		})
	}

	/// Negates a ciphertext.
	pub fn negate(
		&self,
		a: &PyCiphertext,
	) -> PyResult<PyCiphertext> {
		let negated = self.inner.negate(&a.inner).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to negate ciphertext: {:?}",
				e
			))
		})?;
		Ok(PyCiphertext {
			inner: negated,
		})
	}

	/// Adds two ciphertexts.
	pub fn add(
		&self,
		a: &PyCiphertext,
		b: &PyCiphertext,
	) -> PyResult<PyCiphertext> {
		let sum = self.inner.add(&a.inner, &b.inner).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to add ciphertexts: {:?}",
				e
			))
		})?;
		Ok(PyCiphertext {
			inner: sum,
		})
	}

	/// Adds many ciphertexts.
	pub fn add_many(
		&self,
		a: Vec<PyCiphertext>,
	) -> PyResult<PyCiphertext> {
		let mut ciphertexts = Vec::new();
		for c in a {
			ciphertexts.push(c.inner);
		}
		let sum = self.inner.add_many(&ciphertexts).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to add many ciphertexts: {:?}",
				e
			))
		})?;
		Ok(PyCiphertext {
			inner: sum,
		})
	}

	/// Multiplies two ciphertexts.
	pub fn multiply(
		&self,
		a: &PyCiphertext,
		b: &PyCiphertext,
	) -> PyResult<PyCiphertext> {
		let product = self.inner.multiply(&a.inner, &b.inner).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to multiply ciphertexts: {:?}",
				e
			))
		})?;
		Ok(PyCiphertext {
			inner: product,
		})
	}

	/// Multiplies many ciphertexts.
	pub fn multiply_many(
		&self,
		a: Vec<PyCiphertext>,
		relin_keys: &PyRelinearizationKey,
	) -> PyResult<PyCiphertext> {
		let mut ciphertexts = Vec::new();
		for c in a {
			ciphertexts.push(c.inner);
		}
		let product = self
			.inner
			.multiply_many(&ciphertexts, &relin_keys.inner)
			.map_err(|e| {
				PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
					"Failed to multiply many ciphertexts: {:?}",
					e
				))
			})?;
		Ok(PyCiphertext {
			inner: product,
		})
	}

	/// Subtracts two ciphertexts.
	pub fn sub(
		&self,
		a: &PyCiphertext,
		b: &PyCiphertext,
	) -> PyResult<PyCiphertext> {
		let difference = self.inner.sub(&a.inner, &b.inner).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to subtract ciphertexts: {:?}",
				e
			))
		})?;
		Ok(PyCiphertext {
			inner: difference,
		})
	}

	/// Adds a ciphertext and a plaintext.
	pub fn add_plain(
		&self,
		a: &PyCiphertext,
		b: &PyPlaintext,
	) -> PyResult<PyCiphertext> {
		let sum = self.inner.add_plain(&a.inner, &b.inner).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to add ciphertext and plaintext: {:?}",
				e
			))
		})?;
		Ok(PyCiphertext {
			inner: sum,
		})
	}

	/// Subtracts a plaintext from a ciphertext.
	pub fn sub_plain(
		&self,
		a: &PyCiphertext,
		b: &PyPlaintext,
	) -> PyResult<PyCiphertext> {
		let difference = self.inner.sub_plain(&a.inner, &b.inner).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to subtract plaintext from ciphertext: {:?}",
				e
			))
		})?;
		Ok(PyCiphertext {
			inner: difference,
		})
	}

	/// Multiplies a ciphertext by a plaintext.
	pub fn multiply_plain(
		&self,
		a: &PyCiphertext,
		b: &PyPlaintext,
	) -> PyResult<PyCiphertext> {
		let product = self.inner.multiply_plain(&a.inner, &b.inner).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to multiply ciphertext by plaintext: {:?}",
				e
			))
		})?;
		Ok(PyCiphertext {
			inner: product,
		})
	}

	/// Relinearizes a ciphertext.
	pub fn relinearize(
		&self,
		a: &PyCiphertext,
		relin_keys: &PyRelinearizationKey,
	) -> PyResult<PyCiphertext> {
		let relinearized = self
			.inner
			.relinearize(&a.inner, &relin_keys.inner)
			.map_err(|e| {
				PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
					"Failed to relinearize ciphertext: {:?}",
					e
				))
			})?;
		Ok(PyCiphertext {
			inner: relinearized,
		})
	}

	pub fn rotate_rows(
		&self,
		a: &PyCiphertext,
		steps: i32,
		galois_keys: &PyGaloisKey,
	) -> PyResult<PyCiphertext> {
		let rotated = self
			.inner
			.rotate_rows(&a.inner, steps, &galois_keys.inner)
			.map_err(|e| {
				PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
					"Failed to rotate rows: {:?}",
					e
				))
			})?;
		Ok(PyCiphertext {
			inner: rotated,
		})
	}
}
