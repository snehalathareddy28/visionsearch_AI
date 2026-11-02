import io

from fastapi.testclient import TestClient
from PIL import Image

from visionsearch_web_ui import api, sharing


def _image_bytes(image_format="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), color="purple").save(buffer, format=image_format)
    return buffer.getvalue()


def _configure_index(monkeypatch, paths, root):
    monkeypatch.setattr(api, "ROOT", root)
    monkeypatch.setattr(sharing, "ROOT", root)
    monkeypatch.setattr(api, "get_index", lambda: (None, [str(path) for path in paths]))
    monkeypatch.setattr(api, "_share_image_paths_snapshot", None)
    monkeypatch.setattr(api, "_share_image_paths", {})


def test_share_endpoint_serves_only_indexed_supported_images(tmp_path, monkeypatch):
    image_path = tmp_path / "indexed.png"
    image_path.write_bytes(_image_bytes())
    _configure_index(monkeypatch, [image_path], tmp_path)
    monkeypatch.setattr(api, "PUBLIC_SHARING_ENABLED", True)
    image_id = sharing.image_id_for_path(image_path)

    response = TestClient(api.app).get(f"/api/share/images/{image_id}")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.content == image_path.read_bytes()


def test_share_endpoint_rejects_disabled_invalid_and_missing_images(tmp_path, monkeypatch):
    image_path = tmp_path / "indexed.png"
    image_path.write_bytes(_image_bytes())
    _configure_index(monkeypatch, [image_path], tmp_path)
    client = TestClient(api.app)
    image_id = sharing.image_id_for_path(image_path)

    monkeypatch.setattr(api, "PUBLIC_SHARING_ENABLED", False)
    assert client.get(f"/api/share/images/{image_id}").status_code == 404

    monkeypatch.setattr(api, "PUBLIC_SHARING_ENABLED", True)
    assert client.get("/api/share/images/not-an-image-id").status_code == 404
    assert client.get("/api/share/images/..%2F..%2Fsecret").status_code == 404

    image_path.unlink()
    assert client.get(f"/api/share/images/{image_id}").status_code == 404


def test_share_endpoint_does_not_serve_unindexed_or_unsupported_images(tmp_path, monkeypatch):
    indexed_path = tmp_path / "indexed.png"
    indexed_path.write_bytes(_image_bytes())
    unindexed_path = tmp_path / "unindexed.png"
    unindexed_path.write_bytes(_image_bytes())
    unsupported_path = tmp_path / "unsupported.gif"
    unsupported_path.write_bytes(_image_bytes())
    _configure_index(monkeypatch, [indexed_path], tmp_path)
    monkeypatch.setattr(api, "PUBLIC_SHARING_ENABLED", True)
    client = TestClient(api.app)

    assert client.get(
        f"/api/share/images/{sharing.image_id_for_path(unindexed_path)}"
    ).status_code == 404
    assert sharing.image_id_for_path(unsupported_path) is None


def test_normalized_results_use_opaque_ids_without_absolute_paths(tmp_path, monkeypatch):
    image_path = tmp_path / "dataset" / "sample image#1.jpg"
    image_path.parent.mkdir()
    image_path.write_bytes(_image_bytes("JPEG"))
    monkeypatch.setattr(api, "ROOT", tmp_path)
    monkeypatch.setattr(sharing, "ROOT", tmp_path)
    monkeypatch.setattr(api, "PUBLIC_SHARING_ENABLED", False)

    result = api.normalize_results([{"image_path": str(image_path), "score": 0.75}])[0]

    assert result["share_image_id"] == sharing.image_id_for_path(image_path)
    assert result["image_url"] == "/image/dataset/sample%20image%231.jpg"
    assert result["share_url"] is None
    assert str(tmp_path) not in str(result)

    monkeypatch.setattr(api, "PUBLIC_SHARING_ENABLED", True)
    public_result = api.normalize_results([{"image_path": str(image_path)}])[0]
    assert public_result["share_url"] == f"/api/share/images/{result['share_image_id']}"


def test_share_frontend_assets_are_served():
    client = TestClient(api.app)

    page = client.get("/")
    script = client.get("/share.js")
    stylesheet = client.get("/share.css")
    region_script = client.get("/region_search.js")

    assert page.status_code == 200
    assert "region_search.js?v=20261102-2" in page.text
    assert '/share.js' in page.text
    assert script.status_code == 200
    assert "navigator.canShare({ files })" in script.text
    assert stylesheet.status_code == 200
    assert region_script.status_code == 200
    assert "item.parent_path.split" not in region_script.text
    assert 'typeof item.name === "string"' in region_script.text
