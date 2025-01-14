use crate::{
	ciphertext::PyCiphertext,
	context::PyContext,
	keys::{PyPublicKey, PyRelinearizationKey, PySecretKey},
	plaintext::PyPlaintext,
	PyEvaluator,
};
use pyo3::prelude::*;
use sealy::{EvaluatorOps, FromChunk, ToChunk};

#[derive(Debug, Clone)]
#[pyclass(module = "sealy", name = "PlaintextTensor")]
pub struct PyPlaintextTensor {
	inner: sealy::Tensor<sealy::Plaintext>,
}

#[pymethods]
impl PyPlaintextTensor {
	/// Creates a new batch array.
	#[new]
	fn new(ndarr: Vec<PyPlaintext>) -> PyResult<Self> {
		let batch = sealy::Tensor(ndarr.iter().map(|x| x.inner.clone()).collect());
		Ok(Self {
			inner: batch,
		})
	}
}

#[derive(Debug, Clone)]
#[pyclass(module = "sealy", name = "CiphertextTensor")]
pub struct PyCiphertextTensor {
	inner: sealy::Tensor<sealy::Ciphertext>,
}

#[pymethods]
impl PyCiphertextTensor {
	/// Creates a new batch array.
	#[new]
	fn new(ndarr: Vec<PyCiphertext>) -> PyResult<Self> {
		let batch = sealy::Tensor(ndarr.iter().map(|x| x.inner.clone()).collect());
		Ok(Self {
			inner: batch,
		})
	}

	/// Converts the batch array to a list of byte arrays.
	pub fn to_bytes_chunk(&self) -> PyResult<Vec<Vec<u8>>> {
		let bytes = self.inner.to_chunk().map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to get ciphertext batch as bytes: {:?}",
				e
			))
		})?;
		Ok(bytes)
	}

	/// Creates a new ciphertext batch array from a list of byte arrays.
	#[staticmethod]
	pub fn from_bytes_chunk(
		ctx: &PyContext,
		bytes: Vec<Vec<u8>>,
	) -> PyResult<Self> {
		let batch = sealy::Tensor::from_chunk(&ctx.inner, &bytes).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to create ciphertext batch from bytes: {:?}",
				e
			))
		})?;
		Ok(Self {
			inner: batch,
		})
	}
}

/// Encryptor that can encrypt multiple messages at once.
#[pyclass(module = "sealy", name = "TensorEncryptor")]
pub struct PyTensorEncryptor {
	inner: sealy::TensorEncryptor<sealy::Asym>,
}

#[pymethods]
impl PyTensorEncryptor {
	/// Creates a new TensorEncryptor instance with a public key.
	#[new]
	fn new(
		ctx: &PyContext,
		pk: &PyPublicKey,
	) -> PyResult<Self> {
		let ctx = &ctx.inner;
		let pk = &pk.inner;
		let inner = sealy::TensorEncryptor::with_public_key(ctx, pk).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to create TensorEncryptor: {:?}",
				e
			))
		})?;
		Ok(Self {
			inner,
		})
	}

	/// Encrypts a plaintext with the public key and returns the ciphertext as
	/// a serializable object.
	pub fn encrypt(
		&self,
		plaintext: PyPlaintextTensor,
	) -> PyResult<PyCiphertextTensor> {
		let ciphertext = self.inner.encrypt(&plaintext.inner).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to encrypt batch: {:?}",
				e
			))
		})?;
		Ok(PyCiphertextTensor {
			inner: ciphertext,
		})
	}
}

/// Decrypts batches of ciphertexts.
#[pyclass(module = "sealy", name = "TensorDecryptor")]
pub struct PyTensorDecryptor {
	inner: sealy::TensorDecryptor,
}

#[pymethods]
impl PyTensorDecryptor {
	/// Creates a new batch decryptor.
	#[new]
	pub fn new(
		ctx: &PyContext,
		secret_key: &PySecretKey,
	) -> PyResult<Self> {
		let inner = sealy::TensorDecryptor::new(&ctx.inner, &secret_key.inner).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to create TensorDecryptor: {:?}",
				e
			))
		})?;
		Ok(Self {
			inner,
		})
	}

	/// Decrypts a ciphertext and returns the plaintext.
	pub fn decrypt(
		&self,
		ciphertext_batch: &PyCiphertextTensor,
	) -> PyResult<PyPlaintextTensor> {
		let plaintext = self.inner.decrypt(&ciphertext_batch.inner).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to decrypt batch: {:?}",
				e
			))
		})?;
		Ok(PyPlaintextTensor {
			inner: plaintext,
		})
	}
}

/// An encoder that encodes data in batches.
#[pyclass(module = "sealy", name = "CKKSTensorEncoder")]
pub struct PyCKKSTensorEncoder {
	inner: sealy::TensorEncoder<sealy::CKKSEncoder>,
}

#[pymethods]
impl PyCKKSTensorEncoder {
	/// Creates a new TensorEncoder.
	#[new]
	fn new(
		ctx: &PyContext,
		scale: f64,
	) -> PyResult<Self> {
		let encoder = sealy::CKKSEncoder::new(&ctx.inner, scale).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to create CKKSEncoder: {:?}",
				e
			))
		})?;
		let inner = sealy::TensorEncoder::new(encoder);
		Ok(Self {
			inner,
		})
	}

	/// Returns the number of slots in this encoder produces.
	fn get_slot_count(&self) -> usize {
		self.inner.get_slot_count()
	}

	/// Encodes the given data into a plaintext.
	///
	/// # Arguments
	/// * `data` - The data to encode.
	///
	/// # Returns
	/// The encoded plaintext.
	fn encode_float(
		&self,
		data: Vec<f64>,
	) -> PyResult<PyPlaintextTensor> {
		let batch = self.inner.encode_f64(&data).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to encode batch: {:?}",
				e
			))
		})?;
		Ok(PyPlaintextTensor {
			inner: batch,
		})
	}

	/// Decodes the given plaintext into data.
	///
	/// # Arguments
	/// * `batch` - The encoded data.
	///
	/// # Returns
	/// The decoded data.
	fn decode_float(
		&self,
		batch: PyPlaintextTensor,
	) -> PyResult<Vec<f64>> {
		let data = self.inner.decode_f64(&batch.inner).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to decode batch: {:?}",
				e
			))
		})?;

		Ok(data)
	}
}

/// An encoder that encodes data in batches.
#[pyclass(module = "sealy", name = "BFVTensorEncoder")]
pub struct PyBFVTensorEncoder {
	inner: sealy::TensorEncoder<sealy::BFVEncoder>,
}

#[pymethods]
impl PyBFVTensorEncoder {
	/// Creates a new TensorEncoder.
	#[new]
	fn new(ctx: &PyContext) -> PyResult<Self> {
		let encoder = sealy::BFVEncoder::new(&ctx.inner).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to create CKKSEncoder: {:?}",
				e
			))
		})?;
		let inner = sealy::TensorEncoder::new(encoder);
		Ok(Self {
			inner,
		})
	}

	/// Returns the number of slots in this encoder produces.
	fn get_slot_count(&self) -> usize {
		self.inner.get_slot_count()
	}

	/// Encodes the given data into a plaintext.
	///
	/// # Arguments
	/// * `data` - The data to encode.
	///
	/// # Returns
	/// The encoded plaintext.
	fn encode_int(
		&self,
		data: Vec<i64>,
	) -> PyResult<PyPlaintextTensor> {
		let batch = self.inner.encode_i64(&data).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to encode batch: {:?}",
				e
			))
		})?;
		Ok(PyPlaintextTensor {
			inner: batch,
		})
	}

	/// Decodes the given plaintext into data.
	///
	/// # Arguments
	/// * `batch` - The encoded data.
	///
	/// # Returns
	/// The decoded data.
	fn decode_int(
		&self,
		batch: PyPlaintextTensor,
	) -> PyResult<Vec<i64>> {
		let data = self.inner.decode_i64(&batch.inner).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to decode batch: {:?}",
				e
			))
		})?;

		Ok(data)
	}
}

// An evaluator that evaluates batches of data.
#[pyclass(module = "sealy", name = "CKKSTensorEvaluator")]
pub struct PyTensorEvaluator {
	inner: sealy::TensorEvaluator,
}

#[pymethods]
impl PyTensorEvaluator {
	/// Creates a new TensorEvaluator.
	#[new]
	fn new(ctx: &PyContext) -> PyResult<Self> {
		let evaluator = PyEvaluator::new(ctx)?;

		let inner = sealy::TensorEvaluator::new(evaluator.inner);

		Ok(Self {
			inner,
		})
	}

	/// Negates a batch of ciphertexts.
	pub fn negate(
		&self,
		a: &PyCiphertextTensor,
	) -> PyResult<PyCiphertextTensor> {
		let negated = self.inner.negate(&a.inner).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to negate batch: {:?}",
				e
			))
		})?;
		Ok(PyCiphertextTensor {
			inner: negated,
		})
	}

	/// Adds two ciphertexts.
	pub fn add(
		&self,
		a: &PyCiphertextTensor,
		b: &PyCiphertextTensor,
	) -> PyResult<PyCiphertextTensor> {
		let sum = self.inner.add(&a.inner, &b.inner).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to add ciphertexts: {:?}",
				e
			))
		})?;
		Ok(PyCiphertextTensor {
			inner: sum,
		})
	}

	/// Adds many ciphertexts.
	pub fn add_many(
		&self,
		a: Vec<PyCiphertextTensor>,
	) -> PyResult<PyCiphertextTensor> {
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
		Ok(PyCiphertextTensor {
			inner: sum,
		})
	}

	/// Multiplies two ciphertexts.
	pub fn multiply(
		&self,
		a: &PyCiphertextTensor,
		b: &PyCiphertextTensor,
	) -> PyResult<PyCiphertextTensor> {
		let product = self.inner.multiply(&a.inner, &b.inner).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to multiply ciphertexts: {:?}",
				e
			))
		})?;
		Ok(PyCiphertextTensor {
			inner: product,
		})
	}

	/// Multiplies many ciphertexts.
	pub fn multiply_many(
		&self,
		a: Vec<PyCiphertextTensor>,
		relin_keys: &PyRelinearizationKey,
	) -> PyResult<PyCiphertextTensor> {
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
		Ok(PyCiphertextTensor {
			inner: product,
		})
	}

	/// Subtracts two ciphertexts.
	pub fn sub(
		&self,
		a: &PyCiphertextTensor,
		b: &PyCiphertextTensor,
	) -> PyResult<PyCiphertextTensor> {
		let difference = self.inner.sub(&a.inner, &b.inner).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to subtract ciphertexts: {:?}",
				e
			))
		})?;
		Ok(PyCiphertextTensor {
			inner: difference,
		})
	}

	/// Adds a ciphertext and a plaintext.
	pub fn add_plain(
		&self,
		a: &PyCiphertextTensor,
		b: &PyPlaintextTensor,
	) -> PyResult<PyCiphertextTensor> {
		let sum = self.inner.add_plain(&a.inner, &b.inner).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to add plaintext to ciphertext: {:?}",
				e
			))
		})?;
		Ok(PyCiphertextTensor {
			inner: sum,
		})
	}

	/// Subtracts a plaintext from a ciphertext.
	pub fn sub_plain(
		&self,
		a: &PyCiphertextTensor,
		b: &PyPlaintextTensor,
	) -> PyResult<PyCiphertextTensor> {
		let difference = self.inner.sub_plain(&a.inner, &b.inner).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to subtract plaintext from ciphertext: {:?}",
				e
			))
		})?;
		Ok(PyCiphertextTensor {
			inner: difference,
		})
	}

	/// Multiplies a ciphertext by a plaintext.
	pub fn multiply_plain(
		&self,
		a: &PyCiphertextTensor,
		b: &PyPlaintextTensor,
	) -> PyResult<PyCiphertextTensor> {
		let product = self.inner.multiply_plain(&a.inner, &b.inner).map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
				"Failed to multiply ciphertext by plaintext: {:?}",
				e
			))
		})?;
		Ok(PyCiphertextTensor {
			inner: product,
		})
	}

	pub fn relinearize(
		&self,
		a: &PyCiphertextTensor,
		relin_keys: &PyRelinearizationKey,
	) -> PyResult<PyCiphertextTensor> {
		let relinearized = self
			.inner
			.relinearize(&a.inner, &relin_keys.inner)
			.map_err(|e| {
				PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
					"Failed to relinearize ciphertext: {:?}",
					e
				))
			})?;
		Ok(PyCiphertextTensor {
			inner: relinearized,
		})
	}
}
