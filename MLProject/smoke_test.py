"""Wait for the model container and verify one real prediction."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def wait_until_ready(base_url: str, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/ping", timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(2)
    raise TimeoutError(
        f"Model server did not become ready within {timeout_seconds}s: {last_error}"
    )


def predict_one(base_url: str, data_path: Path) -> object:
    X_test = pd.read_csv(data_path / "X_test.csv")
    if X_test.shape[1] != 30:
        raise ValueError(f"Expected 30 features, got {X_test.shape[1]}")

    payload = {
        "dataframe_split": {
            "columns": list(X_test.columns),
            "data": [X_test.iloc[0].tolist()],
        }
    }
    request = urllib.request.Request(
        f"{base_url}/invocations",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.loads(response.read().decode("utf-8"))
        if response.status != 200:
            raise RuntimeError(f"Prediction returned HTTP {response.status}: {body}")

    predictions = body.get("predictions", body) if isinstance(body, dict) else body
    if not isinstance(predictions, list) or len(predictions) != 1:
        raise ValueError(f"Unexpected prediction response: {body}")
    if predictions[0] not in (0, 1):
        raise ValueError(f"Prediction is not a binary class: {predictions[0]}")
    return predictions[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:5001")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--data_path",
        type=Path,
        default=SCRIPT_DIR / "telco_customer_churn_preprocessing",
    )
    args = parser.parse_args()

    wait_until_ready(args.url.rstrip("/"), args.timeout)
    prediction = predict_one(args.url.rstrip("/"), args.data_path)
    print(f"Model smoke test passed. Prediction: {prediction}")


if __name__ == "__main__":
    main()
