
import os
import sys
import requests
from dotenv import load_dotenv
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
SKIP = "\033[93m[SKIP]\033[0m"


def test_groq():
    print("\n--- Test 1: Groq API Key ---")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(f"{FAIL} GROQ_API_KEY not set in .env")
        return False

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            timeout=30.0,
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        )
        reply = response.choices[0].message.content.strip()
        print(f"{PASS} Groq connected. Response: {reply}")
        return True
    except Exception as e:
        print(f"{FAIL} Groq error: {e}")
        return False


def test_supabase():
    print("\n--- Test 2: Supabase Connection ---")
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or url.startswith("your_"):
        print(f"{SKIP} SUPABASE_URL not configured in .env yet")
        print("   -> Go to: Supabase Dashboard > Project Settings > API")
        print("   -> Copy 'Project URL' into SUPABASE_URL in your .env")
        return None
    if not key or key.startswith("your_"):
        print(f"{SKIP} SUPABASE_KEY not configured in .env yet")
        print("   -> Go to: Supabase Dashboard > Project Settings > API")
        print("   -> Copy 'anon public' key into SUPABASE_KEY in your .env")
        return None

    try:
        # Test the REST API with a simple table query (works with anon key)
        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
        }
        resp = requests.get(f"{url}/rest/v1/test_connection?select=id&limit=1", headers=headers, timeout=10)

        if resp.status_code == 200:
            print(f"{PASS} Supabase connected! REST API is responding.")
            return True
        elif resp.status_code == 401:
            print(f"{FAIL} Supabase auth failed — check your SUPABASE_KEY in .env")
            return False
        else:
            print(f"{FAIL} Supabase returned status {resp.status_code}: {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"{FAIL} Supabase error: {e}")
        return False


def test_supabase_read_write():
    print("\n--- Test 2b: Supabase Read/Write ---")
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or url.startswith("your_") or not key or key.startswith("your_"):
        print(f"{SKIP} Supabase not configured yet, skipping read/write test")
        return None

    try:
        # Test write via REST API directly
        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        data = {"message": "hello from test script"}
        resp = requests.post(
            f"{url}/rest/v1/test_connection",
            headers=headers,
            json=data,
            timeout=10,
        )

        if resp.status_code == 201:
            print(f"{PASS} Supabase write succeeded! Row: {resp.json()}")
        elif resp.status_code == 404 or "does not exist" in resp.text:
            print(f"{SKIP} Table 'test_connection' doesn't exist yet (normal for a new project)")
            print("   -> To test read/write, create a table in Supabase Dashboard:")
            print("   -> Table Editor > New Table > name: test_connection")
            print("   -> Add column: message (type: text)")
            print("   -> Uncheck 'Enable Row Level Security' for testing")
            return None
        else:
            print(f"{FAIL} Supabase write returned {resp.status_code}: {resp.text[:200]}")
            return False

        # Read it back
        resp = requests.get(
            f"{url}/rest/v1/test_connection?select=*&limit=1",
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 200:
            print(f"{PASS} Supabase read succeeded! Row: {resp.json()}")
            return True
        else:
            print(f"{FAIL} Supabase read failed: {resp.status_code}")
            return False
    except Exception as e:
        print(f"{FAIL} Supabase read/write error: {e}")
        return False


def test_render():
    print("\n--- Test 3: Render Backend ---")
    render_url = os.environ.get("RENDER_BACKEND_URL")

    if not render_url:
        print(f"{SKIP} RENDER_BACKEND_URL not set in .env (not deployed yet)")
        print("   -> Once deployed, add your Render URL to .env")
        print("   -> Example: RENDER_BACKEND_URL=https://your-app.onrender.com")
        return None

    # Try common health endpoints
    endpoints_to_try = ["/health", "/api/health", "/ping", "/api/ping", "/"]
    render_url = render_url.rstrip("/")

    for endpoint in endpoints_to_try:
        try:
            resp = requests.get(f"{render_url}{endpoint}", timeout=30, verify=False)
            if resp.status_code == 200:
                print(f"{PASS} Render backend is live at {render_url}{endpoint}")
                print(f"   Response: {resp.text[:200]}")
                return True
        except requests.exceptions.ConnectionError:
            continue
        except Exception:
            continue

    # If none of the health endpoints worked, try the base URL
    try:
        resp = requests.get(render_url, timeout=30, verify=False)
        print(f"{PASS} Render backend responded (status {resp.status_code})")
        return True
    except Exception as e:
        print(f"{FAIL} Render backend unreachable: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("  SERVICE CONNECTIVITY TEST")
    print("=" * 50)

    results = {}
    results["Groq API"] = test_groq()
    results["Supabase Connection"] = test_supabase()
    results["Supabase Read/Write"] = test_supabase_read_write()
    results["Render Backend"] = test_render()

    print("\n" + "=" * 50)
    print("  SUMMARY")
    print("=" * 50)
    for service, result in results.items():
        if result is True:
            status = PASS
        elif result is False:
            status = FAIL
        else:
            status = SKIP
        print(f"  {status} {service}")
    print()

    failures = [k for k, v in results.items() if v is False]
    if failures:
        print(f"Fix the failing services above and re-run this script.")
        sys.exit(1)
    else:
        print("All configured services are working!")
