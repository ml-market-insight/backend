from fastapi.testclient import TestClient
from CODE.api.main import app

client = TestClient(app)

def test_get_stocks():
    response = client.get("/stocks")
    assert response.status_code == 200
    assert "available_stocks" in response.json()

def test_post_selected_stocks():
    payload = {"tickers": ["AAPL", "GOOGL"]}
    response = client.post("/selectedstocks", json=payload)
    assert response.status_code == 200, f"Expected 200 OK, but got {response.status_code}. Response text: {response.text}"


# Exécutez les tests si ce fichier est exécuté directement
if __name__ == "__main__":
    import pytest
    pytest.main()
