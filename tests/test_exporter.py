from __future__ import annotations

import httpx

from parity import Reference, Suite, Target
from parity.exporter import CanaryState, render_metrics, serve
from parity.report_model import ReferenceInfo
from parity.tokenization import TokenizerKit
from tests.fake_endpoint import FakeEndpoint, SuiteReference

FAST = {
    "sampler_looks": [60, 150],
    "claimed_context": 1024,
    "longctx_lengths": [256, 1024],
    "template_kwargs": {},
}


def test_metrics_before_any_run_are_still_valid() -> None:
    text = render_metrics(None, 0, 0)
    assert "parity_runs_total 0" in text and "parity_verdict" not in text


async def test_exposition_after_a_run_and_over_http(toy_kit: TokenizerKit) -> None:
    ep = FakeEndpoint(toy_kit, extra_prompt_tokens=1, seed=1)
    rep = await Suite(probes=["template.token_counts"], options=FAST).arun(
        Target("http://h", 'm"x', transport=ep.transport()),
        Reference(
            toy_kit,
            SuiteReference(toy_kit),
            ReferenceInfo(kind="hf:local", model_id="t", revision="r"),
        ),
    )
    state = CanaryState()
    state.update(rep)
    state.update(None, error=True)
    text = state.text()
    assert 'parity_verdict{model="m\\"x",target="http://h"} 1' in text  # deviation, label escaped
    assert (
        'parity_probe_status{model="m\\"x",target="http://h",probe="template.token_counts"} 1'
        in text
    )
    assert "parity_runs_total 1" in text and "parity_run_errors_total 1" in text
    server = serve(state, 0, "127.0.0.1")
    try:
        port = server.server_address[1]
        r = httpx.get(f"http://127.0.0.1:{port}/metrics")
        assert r.status_code == 200 and "version=0.0.4" in r.headers["content-type"]
        assert r.text == text
        assert httpx.get(f"http://127.0.0.1:{port}/nope").status_code == 404
    finally:
        server.shutdown()
        server.server_close()
