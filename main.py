import json
import os
import sys
import time
import urllib.error
import urllib.request

# Model identifier for Meta Llama 3 8B Instruct on Cloudflare Workers AI
MODEL_ID = "@cf/meta/llama-3-8b-instruct"


def query_llama3_edge(account_id: str, api_token: str, user_prompt: str) -> dict:
    """Sends a chat completion request to Meta Llama 3 running on Cloudflare Workers AI."""
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{MODEL_ID}"

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful and concise AI assistant executing at the edge on Cloudflare Workers AI.",
            },
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 256,
        "temperature": 0.6,
    }

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
        "User-Agent": "Edge-Llama3-Client/1.0",
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    start_time = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as response:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        result = json.loads(response.read().decode("utf-8"))
        result["_elapsed_ms"] = elapsed_ms
        return result


def run_simulation(user_prompt: str):
    """Simulates edge execution when Cloudflare credentials are not provided."""
    print("\n[INFO] Running in mock/dry-run mode (no Cloudflare credentials found).")
    print("[INFO] Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN to call live edge nodes.\n")

    mock_response = {
        "result": {
            "response": (
                f"[Edge Node Mock] Llama 3 processed your prompt: '{user_prompt}'\n"
                "Cloudflare Workers AI enables running models closer to users across "
                "300+ edge locations for reduced latency and enhanced data sovereignty."
            )
        },
        "success": True,
        "errors": [],
        "messages": [],
        "_elapsed_ms": 42.1,
    }
    return mock_response


def main():
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()

    prompt = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Explain edge AI inference in two sentences."
    )

    print("=" * 60)
    print("Cloudflare Workers AI - Meta Llama 3 Inference")
    print(f"Target Model : {MODEL_ID}")
    print(f"User Prompt  : {prompt}")
    print("=" * 60)

    if account_id and api_token:
        print("\n[Edge] Dispatching prompt to Cloudflare Workers AI edge network...")
        try:
            res = query_llama3_edge(account_id, api_token, prompt)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            print(f"[ERROR] Cloudflare API returned HTTP {e.code}: {err_body}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            sys.exit(1)
    else:
        res = run_simulation(prompt)

    # Display model output and edge performance metrics
    if res.get("success"):
        reply = res["result"].get("response", "<empty response>")
        latency = res.get("_elapsed_ms", 0)
        print("\n--- Model Response ---")
        print(reply.strip())
        print("----------------------")
        print(f"Latency: {latency:.2f} ms")
    else:
        print("[ERROR] Inference failed:", res.get("errors"))


if __name__ == "__main__":
    main()
