import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["MACRO_AGENT_OFFLINE"] = "1"

from macro_agent.config import load_config          # noqa: E402
from macro_agent.devdata import generate             # noqa: E402
from macro_agent.pipeline import MacroPipeline       # noqa: E402
from macro_agent.storage import Store                # noqa: E402

D = pd.Timestamp("2022-03-15")


def make_pipe(tmp_path, data, cfg=None, max_release=None, skip=()):
    cfg = cfg or load_config()
    cfg.database_url = f"sqlite:///{tmp_path}/t.db"
    store = Store(cfg.database_url)
    for k, df in data.items():
        if k in skip:
            continue
        sc = cfg.series[k]
        if max_release is not None:
            df = df[df["obs_date"] + pd.Timedelta(days=sc.release_lag_days) <= max_release]
        store.append_raw(k, df, sc.unit, sc.source_url)
    return MacroPipeline(cfg, store)


@pytest.fixture(scope="session")
def data():
    return generate(seed=11)


@pytest.fixture(scope="session")
def pipe(tmp_path_factory, data):
    return make_pipe(tmp_path_factory.mktemp("full"), data)
