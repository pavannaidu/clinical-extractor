from urllib.parse import urljoin

from fastapi.testclient import TestClient

from main import app


def test_info_page_is_served_with_and_without_trailing_slash():
    with TestClient(app) as client:
        for path in ("/info", "/info/"):
            response = client.get(path)

            assert response.status_code == 200
            assert response.url.path == "/info/"
            assert response.headers["content-type"].startswith("text/html")
            assert "<title>Structured AI in Life Sciences</title>" in response.text
            assert response.text.count("<section") == 5
            assert 'id="fullscreen-button"' in response.text
            assert 'id="narrative-rail"' in response.text
            assert 'id="slide-position"' not in response.text
            assert response.text.count('data-section="Options"') == 3
            assert 'data-scaffold-copy="true"' not in response.text
            for asset in ("assets/tokens.css", "deck.css?v=20260914-4", "deck.js"):
                assert f'="./{asset}"' in response.text
                assert client.get(urljoin(str(response.url), f"./{asset}")).status_code == 200


def test_static_directories_redirect_to_trailing_slash_and_preserve_query():
    with TestClient(app) as client:
        response = client.get("/info?view=read", follow_redirects=False)

        assert response.status_code == 307
        assert response.headers["location"] == "http://testserver/info/?view=read"


def test_info_assets_are_served():
    with TestClient(app) as client:
        for path, content_type in (
            ("/info/deck.css", "text/css"),
            ("/info/deck.js", "text/javascript"),
            ("/info/assets/tokens.css", "text/css"),
            ("/info/assets/brand/databricks-symbol-color.svg", "image/svg+xml"),
        ):
            response = client.get(path)

            assert response.status_code == 200
            assert response.headers["content-type"].startswith(content_type)


def test_unknown_frontend_route_still_uses_spa_fallback():
    with TestClient(app) as client:
        response = client.get("/schema-workbench")

    assert response.status_code == 200
    assert '<div id="root"></div>' in response.text
