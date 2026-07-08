#!/usr/bin/env python3
"""
notion_auth.py  –  Set up and verify Notion API access.

Usage:
    ./notion_auth.py setup          # save your integration token
    ./notion_auth.py test           # verify the token can reach the database
    ./notion_auth.py show           # print the token file path (not the token)

Token is stored at:  ~/.config/make7th/notion_token
Environment variable NOTION_TOKEN overrides the file if set.

To get a token:
    1. Go to https://www.notion.so/profile/integrations
    2. Click "New integration", give it a name (e.g. "make7th")
    3. Copy the "Internal Integration Secret"
    4. Run:  ./notion_auth.py setup
    5. Share your Notion database with the integration:
       - Open the database page in Notion
       - Click "..." > "Connections" > find your integration and click "Connect"
"""

import os
import sys
import requests
from pathlib import Path

TOKEN_FILE   = Path.home() / ".config" / "make7th" / "notion_token"
DATABASE_ID  = "23205f1277ae8020b1fff7dd6afc01e2"
NOTION_VER   = "2022-06-28"


def get_token():
    """Return token from env var or token file, or None if not configured."""
    token = os.environ.get("NOTION_TOKEN")
    if token:
        return token.strip()
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    return None


def save_token(token):
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(token.strip())
    TOKEN_FILE.chmod(0o600)
    print(f"Token saved to {TOKEN_FILE}")


def test_token(token):
    """Hit the database endpoint and report success or a clear error."""
    resp = requests.get(
        f"https://api.notion.com/v1/databases/{DATABASE_ID}",
        headers={
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_VER,
        },
    )

    if resp.status_code == 200:
        data = resp.json()
        title = ""
        for part in data.get("title", []):
            title += part.get("plain_text", "")
        print(f"OK  –  Connected to database: \"{title}\"")
        return True

    elif resp.status_code == 401:
        print("FAIL  –  Token rejected (401 Unauthorized). Check the token and try again.")
    elif resp.status_code == 403:
        print("FAIL  –  Access denied (403). Share the database with your integration:")
        print("         Notion database > ... > Connections > [your integration]")
    elif resp.status_code == 404:
        print("FAIL  –  Database not found (404). Confirm the database is shared with your integration.")
    else:
        print(f"FAIL  –  Unexpected response {resp.status_code}: {resp.text[:200]}")

    return False


def cmd_setup():
    print("Paste your Notion Internal Integration Secret and press Enter.")
    print("(It starts with 'ntn_' or 'secret_')")
    try:
        token = input("Token: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        sys.exit(1)

    if not token:
        print("No token entered.")
        sys.exit(1)

    save_token(token)
    print("Testing connection...")
    test_token(token)


def cmd_test():
    token = get_token()
    if not token:
        print(f"No token found.\nRun:  {sys.argv[0]} setup")
        sys.exit(1)
    source = "NOTION_TOKEN env var" if os.environ.get("NOTION_TOKEN") else str(TOKEN_FILE)
    print(f"Using token from: {source}")
    ok = test_token(token)
    sys.exit(0 if ok else 1)


def cmd_show():
    if os.environ.get("NOTION_TOKEN"):
        print("Token source: NOTION_TOKEN environment variable")
    elif TOKEN_FILE.exists():
        print(f"Token file:   {TOKEN_FILE}")
    else:
        print(f"No token configured. Run:  {sys.argv[0]} setup")
        sys.exit(1)


COMMANDS = {"setup": cmd_setup, "test": cmd_test, "show": cmd_show}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: {sys.argv[0]} setup|test|show")
        sys.exit(1)
    COMMANDS[sys.argv[1]]()
