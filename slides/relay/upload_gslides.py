"""Upload relay.pptx to Google Drive as a native Google Slides presentation (private to the owner).

    python slides/relay/upload_gslides.py login            # prints the Google login URL
    python slides/relay/upload_gslides.py finish           # reads the redirect URL from stdin
    python slides/relay/upload_gslides.py upload [file]    # -> prints the Google Slides link
    python slides/relay/upload_gslides.py pdf OUT.pdf      # export the last upload as PDF (for checking)

Credentials live outside the repository, in ~/.config/relay-slides/ (see README.txt there):
client_secret.json is the OAuth "Desktop app" client downloaded from the Google Cloud console,
token.json is written after the login. The only scope requested is drive.file: this script can see
and change the files it created itself, and nothing else in the Drive.

Login is two steps so that it also works when the browser runs on another machine: after approving,
Google redirects to http://127.0.0.1:<port>/?code=... . That page fails to load (nothing listens
there); copy its full address from the address bar and pass it to `finish`.

Standard library only.
"""
import base64
import hashlib
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONF = Path(os.environ.get("RELAY_SLIDES_CONF", Path.home() / ".config" / "relay-slides"))
CLIENT, TOKEN, PENDING, STATE = (CONF / n for n in ("client_secret.json", "token.json", "pending.json", "state.json"))
SCOPE = "https://www.googleapis.com/auth/drive.file"
REDIRECT = "http://127.0.0.1:8765/"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
SLIDES_MIME = "application/vnd.google-apps.presentation"


def write_private(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1)


def client():
    if not CLIENT.exists():
        raise SystemExit(f"missing {CLIENT}: download the OAuth client JSON (Desktop app) and save it there")
    c = json.loads(CLIENT.read_text())
    c = c.get("installed") or c.get("web") or c
    if "PASTE_" in c.get("client_id", "PASTE_") or "PASTE_" in c.get("client_secret", "PASTE_"):
        raise SystemExit(f"{CLIENT} still contains the PASTE_ placeholders")
    return c


def post_form(url, fields):
    req = urllib.request.Request(url, data=urllib.parse.urlencode(fields).encode())
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Google answered {e.code}: {e.read().decode(errors='replace')[:400]}")


def login():
    c = client()
    verifier = secrets.token_urlsafe(64)
    state = secrets.token_urlsafe(16)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    write_private(PENDING, {"verifier": verifier, "state": state, "created": time.time()})
    q = {"client_id": c["client_id"], "redirect_uri": REDIRECT, "response_type": "code", "scope": SCOPE,
         "access_type": "offline", "prompt": "consent", "state": state,
         "code_challenge": challenge, "code_challenge_method": "S256"}
    print("Open this URL in your browser and approve access:\n")
    print(c.get("auth_uri", "https://accounts.google.com/o/oauth2/auth") + "?" + urllib.parse.urlencode(q))
    print(f"\nAfterwards the browser lands on {REDIRECT}?code=... (the page will not load).")
    print("Copy that full address and run:  upload_gslides.py finish   (paste it on stdin)")


def finish(arg=None):
    c = client()
    if not PENDING.exists():
        raise SystemExit("no login in progress: run `login` first")
    pending = json.loads(PENDING.read_text())
    raw = (Path(arg).read_text() if arg and Path(arg).exists() else arg or sys.stdin.read()).strip()
    q = urllib.parse.parse_qs(urllib.parse.urlparse(raw).query) if "?" in raw else {"code": [raw]}
    if "error" in q:
        raise SystemExit(f"Google reported: {q['error'][0]}")
    if q.get("state", [pending["state"]])[0] != pending["state"]:
        raise SystemExit("state mismatch: this redirect URL belongs to a different login attempt; run `login` again")
    tok = post_form(c.get("token_uri", "https://oauth2.googleapis.com/token"), {
        "code": q["code"][0], "client_id": c["client_id"], "client_secret": c["client_secret"],
        "redirect_uri": REDIRECT, "grant_type": "authorization_code", "code_verifier": pending["verifier"]})
    tok["expires_at"] = time.time() + tok.get("expires_in", 0) - 60
    write_private(TOKEN, tok)
    PENDING.unlink()
    print(f"logged in; token saved to {TOKEN} (scope: {tok.get('scope')})")


def access_token():
    if not TOKEN.exists():
        raise SystemExit("not logged in: run `login`, then `finish`")
    tok = json.loads(TOKEN.read_text())
    if time.time() >= tok.get("expires_at", 0):
        if "refresh_token" not in tok:
            raise SystemExit("token expired and there is no refresh token: run `login` again")
        c = client()
        new = post_form(c.get("token_uri", "https://oauth2.googleapis.com/token"), {
            "client_id": c["client_id"], "client_secret": c["client_secret"],
            "refresh_token": tok["refresh_token"], "grant_type": "refresh_token"})
        tok.update(new)
        tok["expires_at"] = time.time() + new.get("expires_in", 0) - 60
        write_private(TOKEN, tok)
    return tok["access_token"]


def api(method, url, body=None, headers=None):
    h = {"Authorization": "Bearer " + access_token()}
    h.update(headers or {})
    req = urllib.request.Request(url, data=body, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Drive API {method} failed with {e.code}: {e.read().decode(errors='replace')[:600]}")


def upload(path=None, title=None):
    path = Path(path) if path else HERE / "relay.pptx"
    title = title or "Relay: let the partners pass the message " + time.strftime("(%Y-%m-%d %H:%M)")
    boundary = "relay" + secrets.token_hex(12)
    meta = json.dumps({"name": title, "mimeType": SLIDES_MIME}).encode()   # the Slides MIME type makes Drive convert
    body = b"".join([
        f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n".encode(), meta,
        f"\r\n--{boundary}\r\nContent-Type: {PPTX_MIME}\r\n\r\n".encode(), path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode()])
    out = json.loads(api("POST", "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id,name,mimeType,webViewLink",
                         body, {"Content-Type": f"multipart/related; boundary={boundary}"}))
    write_private(STATE, {"last": out, "uploaded": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": str(path)})
    print(f"created '{out['name']}' ({out['mimeType']})\n{out['webViewLink']}")
    return out


def export_pdf(dest):
    if not STATE.exists():
        raise SystemExit("nothing uploaded yet")
    fid = json.loads(STATE.read_text())["last"]["id"]
    data = api("GET", f"https://www.googleapis.com/drive/v3/files/{fid}/export?mimeType=application/pdf")
    Path(dest).write_bytes(data)
    print(f"wrote {dest} ({len(data) / 1e6:.1f} MB)")


def main():
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if cmd == "login":
        login()
    elif cmd == "finish":
        finish(*args[:1])
    elif cmd == "upload":
        upload(*args[:2])
    elif cmd == "pdf" and args:
        export_pdf(args[0])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
