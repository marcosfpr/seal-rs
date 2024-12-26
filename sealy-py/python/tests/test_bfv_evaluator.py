from typing import Tuple

from sealy import (BFVEncoder, BfvEncryptionParametersBuilder, Evaluator,
                   CoefficientModulus, Context, Decryptor, DegreeType,
                   Encryptor, KeyGenerator, PlainModulus, SecurityLevel)


def run_bfv_test() -> (
    Tuple[
        "Decryptor", "BFVEncoder", "Encryptor", "Evaluator", "KeyGenerator"
    ]
):
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

    return (decryptor, encoder, encryptor, evaluator, gen)


def make_vec(encoder):
    return [
        encoder.get_slot_count() // 2 - i
        for i in range(encoder.get_slot_count())
    ]


def make_small_vec(encoder):
    return [16 - i % 32 for i in range(encoder.get_slot_count())]


def test_can_create_and_destroy_evaluator():
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
    evaluator = Evaluator(ctx)
    del evaluator


def test_can_negate():
    decryptor, encoder, encryptor, evaluator, _ = run_bfv_test()

    vec = make_vec(encoder)
    encoded = encoder.encode_int(vec)
    encrypted = encryptor.encrypt(encoded)

    result = evaluator.negate(encrypted)

    decrypted = decryptor.decrypt(result)

    decoded = encoder.decode_int(decrypted)

    assert len(vec) == len(decoded)
    for i in range(len(vec)):
        assert vec[i] == -decoded[i]


def test_can_add():
    decryptor, encoder, encryptor, evaluator, _ = run_bfv_test()

    a = make_vec(encoder)
    b = make_vec(encoder)

    encoded_a = encoder.encode_int(a)
    encoded_b = encoder.encode_int(b)

    encrypted_a = encryptor.encrypt(encoded_a)
    encrypted_b = encryptor.encrypt(encoded_b)

    result = evaluator.add(encrypted_a, encrypted_b)

    decrypted = decryptor.decrypt(result)

    decoded = encoder.decode_int(decrypted)

    assert len(a) == len(decoded)
    for i in range(len(a)):
        assert a[i] + b[i] == decoded[i]


def test_can_sub():
    decryptor, encoder, encryptor, evaluator, _ = run_bfv_test()

    a = make_vec(encoder)
    b = make_vec(encoder)

    encoded_a = encoder.encode_int(a)
    encoded_b = encoder.encode_int(b)

    encrypted_a = encryptor.encrypt(encoded_a)
    encrypted_b = encryptor.encrypt(encoded_b)

    result = evaluator.sub(encrypted_a, encrypted_b)

    decrypted = decryptor.decrypt(result)

    decoded = encoder.decode_int(decrypted)

    assert len(a) == len(decoded)
    for i in range(len(a)):
        assert a[i] - b[i] == decoded[i]


def test_can_multiply():
    decryptor, encoder, encryptor, evaluator, _ = run_bfv_test()

    a = make_small_vec(encoder)
    b = make_small_vec(encoder)

    encoded_a = encoder.encode_int(a)
    encoded_b = encoder.encode_int(b)

    encrypted_a = encryptor.encrypt(encoded_a)
    encrypted_b = encryptor.encrypt(encoded_b)

    result = evaluator.multiply(encrypted_a, encrypted_b)

    decrypted = decryptor.decrypt(result)

    decoded = encoder.decode_int(decrypted)

    assert len(a) == len(decoded)
    for i in range(len(a)):
        assert a[i] * b[i] == decoded[i]
