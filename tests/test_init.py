import oev


def test_package_reexports_oev_class():
    assert oev.OEV is not None
    assert oev.__version__ == "0.4.0"
    from oev.infer import OEV as InferOEV

    assert oev.OEV is InferOEV
