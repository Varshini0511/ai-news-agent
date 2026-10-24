# AI News Agent

Watches AI blogs/feeds, has Claude score each item against your interests (`sources.yaml`),
and pushes only the good ones to Telegram (phone + laptop app) and `digest.md`.

## Setup
1. `pip install -r requirements.txt`
2. Telegram: message @BotFather -> `/newbot` -> copy token. Send your bot any message, then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `chat.id`.
3. Set env vars (PowerShell):
   ```
   $env:ANTHROPIC_API_KEY="..."; $env:TELEGRAM_BOT_TOKEN="..."; $env:TELEGRAM_CHAT_ID="..."
   ```
4. `python agent.py` (once) or `python agent.py --loop 30` (every 30 min).

## True 24/7 (laptop off)
Push this repo to GitHub, add the 3 values as repo Secrets; `.github/workflows/agent.yml`
runs it every 3 hours for free. (Workflow file must live at the repo root's `.github/`,
so if `Agents` is the repo root, move it there.)

## Learn next
- Add a tool-using step: let Claude fetch the full article before scoring.
- Add `/ask` Telegram commands so you can chat with your feed.
- Add sources: YouTube channel RSS, GitHub releases (`.atom`), X via Nitter/RSSHub.
