from flask import Flask, jsonify, render_template_string
from dotenv import load_dotenv
import os
import requests as req
import time

load_dotenv()

app = Flask(__name__)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Service Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container { max-width: 600px; width: 90%; padding: 2rem 0; }
        h1 {
            text-align: center;
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
            color: #f8fafc;
        }
        .subtitle {
            text-align: center;
            color: #94a3b8;
            margin-bottom: 2rem;
            font-size: 0.9rem;
        }
        .card {
            background: #1e293b;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border: 1px solid #334155;
        }
        .card-left { display: flex; align-items: center; gap: 1rem; }
        .icon { font-size: 1.5rem; }
        .service-name { font-weight: 600; font-size: 1rem; }
        .service-detail { color: #94a3b8; font-size: 0.8rem; margin-top: 2px; }
        .badge {
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .badge-pass { background: #064e3b; color: #34d399; }
        .badge-fail { background: #7f1d1d; color: #fca5a5; }
        .badge-skip { background: #78350f; color: #fbbf24; }
        .summary {
            text-align: center;
            margin-top: 2rem;
            padding: 1rem;
            border-radius: 12px;
            font-weight: 600;
        }
        .summary-pass { background: #064e3b; color: #34d399; border: 1px solid #065f46; }
        .summary-fail { background: #7f1d1d; color: #fca5a5; border: 1px solid #991b1b; }
        .refresh {
            display: block;
            margin: 1.5rem auto 0;
            padding: 0.6rem 1.5rem;
            background: #3b82f6;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 500;
        }
        .refresh:hover { background: #2563eb; }
        .timestamp { text-align: center; color: #64748b; font-size: 0.75rem; margin-top: 1rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Service Dashboard</h1>
        <p class="subtitle">Real-time status of all connected services</p>

        {% for s in services %}
        <div class="card">
            <div class="card-left">
                <span class="icon">{{ s.icon }}</span>
                <div>
                    <div class="service-name">{{ s.name }}</div>
                    <div class="service-detail">{{ s.detail }}</div>
                </div>
            </div>
            <span class="badge badge-{{ s.status }}">{{ s.status }}</span>
        </div>
        {% endfor %}

        <div class="summary {{ 'summary-pass' if all_pass else 'summary-fail' }}">
            {{ "All systems operational" if all_pass else "Some services need attention" }}
        </div>

        <button class="refresh" onclick="location.reload()">Refresh Status</button>
        <p class="timestamp">Last checked: {{ timestamp }}</p>
    </div>
</body>
</html>
"""


def check_groq():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {"name": "Groq API", "icon": "🤖", "status": "skip", "detail": "API key not configured"}
    try:
        from openai import OpenAI
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key, timeout=15.0)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Say ok"}],
        )
        return {"name": "Groq API", "icon": "🤖", "status": "pass", "detail": "Connected and responding"}
    except Exception as e:
        return {"name": "Groq API", "icon": "🤖", "status": "fail", "detail": str(e)[:60]}


def check_supabase():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        return {"name": "Supabase Database", "icon": "🗄️", "status": "skip", "detail": "Credentials not configured"}
    try:
        headers = {"apikey": key, "Authorization": f"Bearer {key}"}
        resp = req.get(f"{url}/rest/v1/test_connection?select=id&limit=1", headers=headers, timeout=10)
        if resp.status_code == 200:
            return {"name": "Supabase Database", "icon": "🗄️", "status": "pass", "detail": "Connected — REST API responding"}
        return {"name": "Supabase Database", "icon": "🗄️", "status": "fail", "detail": f"Status {resp.status_code}"}
    except Exception as e:
        return {"name": "Supabase Database", "icon": "🗄️", "status": "fail", "detail": str(e)[:60]}


def check_render():
    return {"name": "Render Backend", "icon": "🚀", "status": "pass", "detail": "You're looking at it right now!"}


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/")
def home():
    services = [check_groq(), check_supabase(), check_render()]
    all_pass = all(s["status"] == "pass" for s in services)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    return render_template_string(DASHBOARD_HTML, services=services, all_pass=all_pass, timestamp=timestamp)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
