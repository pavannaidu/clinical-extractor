from pathlib import Path
import sys

from fastapi.testclient import TestClient


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from main import app  # noqa: E402


def test_info_page_is_served_with_and_without_trailing_slash():
    with TestClient(app) as client:
        for path in ("/info", "/info/"):
            response = client.get(path)

            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/html")
            assert "<title>Structured AI in Life Sciences</title>" in response.text
            assert response.text.count("<section") == 5
            assert 'id="fullscreen-button"' in response.text
            assert 'id="narrative-rail"' in response.text
            assert 'id="slide-position"' not in response.text
            assert response.text.count('data-section="Options"') == 3
            assert 'data-scaffold-copy="true"' not in response.text
            assert 'href="/info/assets/tokens.css"' in response.text
            assert 'href="/info/deck.css?v=20260914-4"' in response.text
            assert 'src="/info/deck.js"' in response.text


def test_info_assets_are_served_for_the_route_without_a_trailing_slash():
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


def test_info_backup_preserves_the_previous_page():
    with TestClient(app) as client:
        for path in ("/info-backup", "/info-backup/"):
            response = client.get(path)

            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/html")
            assert "<title>Structured AI in Life Sciences</title>" in response.text
            assert 'data-page="structured-ai-slide-deck"' in response.text
            assert response.text.count('<section class="slide') == 5
            assert 'data-action="previous"' in response.text
            assert 'data-action="next"' in response.text
            assert '.stage-frame, .stage { width: 1920px; height: 1080px; }' in response.text


def test_unknown_frontend_route_still_uses_spa_fallback():
    with TestClient(app) as client:
        response = client.get("/schema-workbench")

    assert response.status_code == 200
    assert '<div id="root"></div>' in response.text
