"""C1: register, manifest, checksum verification, header dumps, fetch."""

from __future__ import annotations

import hashlib
import http.server
import json
import threading
from functools import partial

import pytest

from scenario_fraud import acquire
from tests.conftest import write_csv


def test_register_copies_checksums_and_records(proj, tmp_path):
    src = write_csv(tmp_path / "dl" / "Medicare PS 2023.csv", ["Rndrng_NPI", "Tot_Benes"], [["1234567893", "12"]])
    e = acquire.register(proj, src, "cms_ps", 2023, "data", source_page="https://example/landing")
    assert e.path == "data/raw/cms_ps/2023/Medicare PS 2023.csv"
    assert e.sha256 == hashlib.sha256(src.read_bytes()).hexdigest()
    assert e.size_bytes == src.stat().st_size and e.file_url == "manual" and e.route == "manual"
    assert src.exists()  # copied, not moved
    [m] = acquire.load_manifest(proj)
    assert m == e


def test_register_is_idempotent_and_refuses_conflicts(proj, tmp_path):
    a = write_csv(tmp_path / "a" / "f.csv", ["x"], [["1"]])
    acquire.register(proj, a, "dme_p", 2024, "data")
    acquire.register(proj, a, "dme_p", 2024, "data")
    assert len(acquire.load_manifest(proj)) == 1
    b = write_csv(tmp_path / "b" / "f.csv", ["x"], [["2"]])
    with pytest.raises(acquire.ManifestError, match="different content"):
        acquire.register(proj, b, "dme_p", 2024, "data")


@pytest.mark.parametrize(
    "source,kind,route,match",
    [("nope", "data", "manual", "unknown source"), ("cms_ps", "snapshot", "manual", "not valid"),
     ("leie", "current", "ftp", "route")],
)
def test_register_validates(proj, tmp_path, source, kind, route, match):
    f = write_csv(tmp_path / "f.csv", ["x"], [["1"]])
    with pytest.raises(acquire.ManifestError, match=match):
        acquire.register(proj, f, source, 2023, kind, route=route)


def test_verify_detects_tampering_and_missing(proj, tmp_path):
    f = write_csv(tmp_path / "f.csv", ["x"], [["1"]])
    e = acquire.register(proj, f, "cms_prov", 2022, "data")
    assert acquire.verify(proj) == []
    p = proj.abs(e.path)
    p.write_bytes(p.read_bytes().replace(b"1", b"2"))
    assert acquire.verify(proj)[0]["problem"] == "sha256 mismatch"
    p.unlink()
    assert acquire.verify(proj)[0]["problem"] == "missing"


def test_header_dump_counts_rows_and_handles_quotes_and_latin1(proj, tmp_path):
    rows = [["1234567893", "Smith, \"Jr\"\nline2"], ["1234567893", "Gómez"]]
    f = write_csv(tmp_path / "f.csv", [" Rndrng_NPI", "Name "], rows, encoding="latin-1")
    e = acquire.register(proj, f, "cms_ps", 2021, "data")
    d = acquire.header_dump(proj, e)
    assert d["columns"] == ["Rndrng_NPI", "Name"]
    assert d["n_rows"] == 2 and d["encoding"] in ("cp1252", "latin-1")
    saved = json.loads(acquire.header_path(proj, e).read_text())
    assert saved["sha256"] == e.sha256


def test_dump_all_headers_skips_documents(proj, tmp_path):
    f = write_csv(tmp_path / "f.csv", ["a"], [["1"], ["2"]])
    pdf = tmp_path / "dd.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    acquire.register(proj, f, "cms_ps", 2020, "data")
    acquire.register(proj, pdf, "cms_ps", 2020, "dictionary")
    dumps = acquire.dump_all_headers(proj)
    assert [d["n_rows"] for d in dumps] == [2]
    assert dumps[0]["row_counter"] == "duckdb"


def test_fetch_streams_and_registers(proj, tmp_path):
    served = tmp_path / "srv"
    write_csv(served / "UPDATED.csv", ["NPI", "EXCLTYPE"], [["0000000000", "1128a1"]])
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(served))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    s = acquire._session()
    s.trust_env = False  # bypass any HTTPS proxy for localhost
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/UPDATED.csv"
        e = acquire.fetch(proj, url, "leie", 2026, "current", route="oig", as_of="2026-10-12", session=s)
        assert e.file_url == url and e.as_of == "2026-10-12"
        assert proj.abs(e.path).read_bytes() == (served / "UPDATED.csv").read_bytes()
        with pytest.raises(acquire.ManifestError, match="HTTP 404"):
            acquire.fetch(proj, url + "x", "leie", 2026, "current", route="oig", session=s)
        assert not list(proj.raw.rglob("*.part"))  # failed download leaves nothing behind
    finally:
        s.close()
        httpd.shutdown()
        httpd.server_close()


def test_compress_keeps_original_checksum(proj, tmp_path):
    from scenario_fraud.harmonise import draft_column_map  # noqa: F401  (import check)

    f = tmp_path / "x.csv"
    f.write_text("Rndrng_NPI,Tot_Srvcs\n1234567893,5\n", encoding="utf-8")
    e = acquire.register(proj, f, "cms_prov", 2020, "data")
    d0 = acquire.header_dump(proj, e)
    z = acquire.compress(proj, e.path)
    assert z.path.endswith(".csv.gz") and not (proj.root / e.path).exists()
    assert acquire.content_sha256(z) == e.sha256 and z.extra["original_size_bytes"] == e.size_bytes
    assert acquire.verify(proj) == []
    assert acquire.header_path(proj, z) == acquire.header_path(proj, e)
    d1 = acquire.header_dump(proj, z)
    assert d1["columns"] == d0["columns"] and d1["sha256"] == e.sha256 and d1["n_rows"] == 1


def test_restore_redownloads_and_checks(proj, tmp_path):
    served = tmp_path / "srv"
    write_csv(served / "a.csv", ["Rndrng_NPI", "Tot_Srvcs"], [["1234567893", "5"]])
    write_csv(served / "UPDATED.csv", ["NPI", "EXCLTYPE"], [["0", "1128a1"]])
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(served))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    s = acquire._session()
    s.trust_env = False
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        a = acquire.fetch(proj, f"{base}/a.csv", "cms_prov", 2020, "data", route="other", session=s)
        z = acquire.compress(proj, a.path)
        u = acquire.fetch(proj, f"{base}/UPDATED.csv", "leie", 2026, "current", route="oig", session=s)
        proj.abs(z.path).unlink()
        proj.abs(u.path).unlink()
        write_csv(served / "UPDATED.csv", ["NPI", "EXCLTYPE"], [["0", "1128b4"]])   # the source replaced its file
        res = {r["path"]: r["status"] for r in acquire.restore(proj, session=s)}
        assert res[z.path] == "restored" and acquire.verify(proj) == [{"path": u.path, "problem": "missing"}]
        assert res[u.path].startswith("checksum mismatch") and not list(proj.raw.rglob("*.part"))
    finally:
        s.close()
        httpd.shutdown()
        httpd.server_close()
