def test_cli_imports_without_heavy_deps():
    # Should import CLI without importing tensorflow/opencv at module import time.
    from liptype_rebuild.cli.entrypoint import main  # noqa: F401

