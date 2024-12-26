[![Crates.io][crates-badge]][crates-url]
[![PyPI][pypi-badge]][pypi-url]
[![CI][ci-badge]][ci-url]

[crates-badge]: https://img.shields.io/crates/v/sealy.svg
[crates-url]: https://crates.io/crates/sealy
[pypi-badge]: https://img.shields.io/pypi/pyversions/sealy
[pypi-url]: https://pypi.org/project/sealy/
[ci-badge]: https://img.shields.io/github/actions/workflow/status/marcosfpr/sealy/pypublish.yml
[ci-url]: https://github.com/marcosfpr/sealy/actions?query=+branch%3Amain

<br />
<p align="center">
  <h3 align="center">seal bindings</h3>

  <p align="center">
    <a href="https://www.microsoft.com/en-us/research/project/microsoft-seal"><strong>Microsoft SEAL bindings for Rust and Python</strong></a>
    <br />
  </p>
</p>

## 🌟 seal

FFI bindings from the famous [SEAL](https://github.com/microsoft/SEAL) library for Rust and Python. 
The main goal of this project is to provide a simple and fast way to install SEAL for both programming languages.

### Built With

The SEAL bindings are a continuation from the [seal_fhe](https://github.com/sunscreen-tech/sunscreen/tree/d9f64f4283b7a4471dd0247b6f5ef769051a649f/seal_fhe) crate, with the support for the CKKS scheme and the addition of new features like tensor encoders, that allow us to overcome the size barriers of the ciphertext tensors and create AI applications easily with high-dimensional encrypted ciphertext.

### Prerequisites

Currently, this crate is available only for a few architectures. Please, make sure that your operating system is compatible with any build that is working:

|    System     |                                                   Support                                                  |
| :-----------: | :--------------------------------------------------------------------------------------------------------: |
| MacOSX aarch6 | [![seal-w64](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/marcosfpr/sealy) |
| Linux x86_64  | [![seal-w64](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/marcosfpr/sealy) |

### Instalation

#### Python

Make sure your OS is supported. If it is, just type:

```sh
pip install sealy
```

If the OS/Platform that you use it is not in the supported list, feel free too try to clone this project and build yourself locally.

#### Rust

```
cargo add sealy
```

### Usage

#### Python

Here is a simple example of multiplying a ciphertext array to a plaintext array.

```python
from sealy import (BFVEncoder, BfvEncryptionParametersBuilder, Evaluator,
                  CoefficientModulus, Context, Decryptor, DegreeType,
                  Encryptor, KeyGenerator, PlainModulus, SecurityLevel)

params = (
    BfvEncryptionParametersBuilder()
    .with_poly_modulus_degree(DegreeType(8192))
    .with_coefficient_modulus(
        CoefficientModulus.create(DegreeType(8192), [50, 30, 30, 50, 50])
    )
    .with_plain_modulus(PlainModulus.batching(DegreeType(8192), 32))
    .build()
)

ctx = Context(params, False, SecurityLevel(128))
gen = KeyGenerator(ctx)

encoder = BFVEncoder(ctx)

public_key = gen.create_public_key()
secret_key = gen.secret_key()

encryptor = Encryptor(ctx, public_key)
decryptor = Decryptor(ctx, secret_key)
evaluator = Evaluator(ctx)

plaintext = [1, 2, 3]
factor = [2, 2, 2]

encoded_plaintext = encoder.encode_int(plaintext)
encoded_factor = encoder.encode_int(factor)

ciphertext = encryptor.encrypt(encoded_plaintext)
ciphertext_result = evaluator.multiply_plain(ciphertext, encoded_factor)

decrypted = decryptor.decrypt(ciphertext_result)
decoded = encoder.decode_int(decrypted)

print(decoded[:3]) # [2, 4, 6]
```

#### Rust

Equivalent code from above's example, written in rust:

```rust
use seal::{
	BFVEncoder, Evaluator, BfvEncryptionParametersBuilder, CoefficientModulus, Context,
	Decryptor, DegreeType, Encoder, Encryptor, Evaluator, KeyGenerator, PlainModulus,
	SecurityLevel,
};

fn main() -> anyhow::Result<()> {
	let params = BfvEncryptionParametersBuilder::new()
		.set_poly_modulus_degree(DegreeType::D8192)
		.set_coefficient_modulus(
			CoefficientModulus::create(DegreeType::D8192, &[50, 30, 30, 50, 50]).unwrap(),
		)
		.set_plain_modulus(PlainModulus::batching(DegreeType::D8192, 32)?)
		.build()?;

	let ctx = Context::new(&params, false, SecurityLevel::TC128)?;
	let gen = KeyGenerator::new(&ctx)?;

	let encoder = BFVEncoder::new(&ctx)?;

	let public_key = gen.create_public_key();
	let secret_key = gen.secret_key();

	let encryptor = Encryptor::with_public_key(&ctx, &public_key)?;
	let decryptor = Decryptor::new(&ctx, &secret_key)?;
	let evaluator = Evaluator::new(&ctx)?;

	let plaintext: Vec<i64> = vec![1, 2, 3];
	let factor = vec![2, 2, 2];

	let encoded_plaintext = encoder.encode_u64(&plaintext)?;
	let encoded_factor = encoder.encode_u64(&factor)?;

	let ciphertext = encryptor.encrypt(&encoded_plaintext)?;
	let ciphertext_result = evaluator.multiply_plain(&ciphertext, &encoded_factor)?;

	let decrypted = decryptor.decrypt(&ciphertext_result)?;
	let decoded = encoder.decode_u64(&decrypted);

	println!("{:?}", &decoded.into_iter().take(3).collect::<Vec<_>>()); // [2, 4, 6]

	Ok(())
}
```

<!-- ROADMAP -->

## Roadmap

The project is in the early stages of development.

See the [open issues](https://github.com/marcosfpr/sealy/issues) for a list of issues and proposed features.

**OBS**: To propose new features or report bugs, check out the correct templates.

<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
