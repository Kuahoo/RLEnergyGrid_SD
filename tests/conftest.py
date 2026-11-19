from pathlib import Path

import pytest

from sdge_microgrid.data.synthetic import generate_year_traces, save_processed


@pytest.fixture(scope="session")
def processed_data_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("processed")
    save_processed(d, generate_year_traces(seed=123))
    return d
