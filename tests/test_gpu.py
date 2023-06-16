import jax


def test_gpu():
    assert jax.devices()[0].platform != "cpu"
