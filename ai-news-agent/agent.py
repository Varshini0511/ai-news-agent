"""AI news agent: fetch feeds -> an LLM (Gemini/Grok/Groq/Claude) scores/summarizes -> push to Telegram + digest.md.

Run once:      python agent.py
Run forever:   python agent.py --loop 30     (every 30 minutes)
"""
import argparse, json, os, sys, time, datetime, pathlib
import difflib, smtplib
from email.message import EmailMessage
import feedparser, requests, yaml
from bs4 import BeautifulSoup

HERE = pathlib.Path(__file__).parent

# Minimal .env loader (KEY=value lines); real environment variables win.
if (HERE / ".env").exists():
    for _l in (HERE / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in _l and not _l.lstrip().startswith("#"):
            _k, _v = _l.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))
os.environ.setdefault("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))  # accept either name
SEEN = HERE / "seen.json"
DIGEST = HERE / "digest.md"
PAGES = HERE / "pages.json"  # last snapshot of each watched page
# Providers tried in order; only those with an API key set are used. Override models with e.g. GEMINI_MODEL.
PROVIDERS = {
    "gemini": ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai", os.getenv("GEMINI_MODEL", "gemini-3.5-flash")),
    "xai": ("XAI_API_KEY", "https://api.x.ai/v1", os.getenv("XAI_MODEL", "grok-4")),
    "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1", os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")),
}
ORDER = os.getenv("LLM_ORDER", "gemini,xai,groq,anthropic").split(",")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
MIN_SCORE = int(os.getenv("MIN_SCORE", "6"))  # 1-10 relevance threshold


def llm(system, user, max_tokens=1500):
    """Ask the first configured provider that works; fall through on error/quota."""
    errors = []
    for name in ORDER:
        try:
            if name == "anthropic":
                if not os.getenv("ANTHROPIC_API_KEY"):
                    continue
                import anthropic
                m = anthropic.Anthropic().messages.create(
                    model=ANTHROPIC_MODEL, max_tokens=max_tokens, system=system,
                    messages=[{"role": "user", "content": user}])
                return m.content[0].text
            keyvar, base, model = PROVIDERS[name]
            if not os.getenv(keyvar):
                continue
            r = requests.post(f"{base}/chat/completions", timeout=90,
                              headers={"Authorization": f"Bearer {os.environ[keyvar]}"},
                              json={"model": model, "max_tokens": max_tokens,
                                    "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": user}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as ex:
            errors.append(f"{name}: {ex}")
    raise RuntimeError("no LLM provider worked: " + "; ".join(errors) if errors else
                       "no API key set (GEMINI_API_KEY / XAI_API_KEY / GROQ_API_KEY / ANTHROPIC_API_KEY)")


def load_seen():
    return set(json.loads(SEEN.read_text())) if SEEN.exists() else set()


def fetch_new(cfg, seen):
    items = []
    for f in cfg["feeds"]:
        try:
            for e in feedparser.parse(f["url"], agent="Mozilla/5.0 ai-news-agent").entries[:15]:
                link = e.get("link")
                if link and link not in seen:
                    items.append({"source": f["name"], "title": e.get("title", ""),
                                  "link": link, "text": (e.get("summary") or "")[:600]})
        except Exception as ex:
            print(f"[warn] {f['name']}: {ex}", file=sys.stderr)
    return items


def rank(interests, items):
    """One LLM call per batch of 25: returns [{id, score, summary}]."""
    out = []
    for i in range(0, len(items), 25):
        batch = items[i:i + 25]
        listing = "\n".join(f"{n}. [{it['source']}] {it['title']} - {it['text'][:300]}"
                            for n, it in enumerate(batch))
        text = llm("You curate an AI news feed for a builder. Reply with ONLY a JSON list: "
                   '[{"id":int,"score":1-10,"summary":"one sentence, why it matters"}] for every item.',
                   f"Interests: {interests}\n\nItems:\n{listing}", 4000)
        try:
            for r in json.loads(text[text.index("["): text.rindex("]") + 1]):
                if r["score"] >= MIN_SCORE and 0 <= r["id"] < len(batch):
                    out.append({**batch[r["id"]], **r})
        except Exception as ex:
            print(f"[warn] bad model output: {ex}", file=sys.stderr)
    return sorted(out, key=lambda x: -x["score"])


def check_pages(cfg):
    """Snapshot watched pages; when text changes, ask the LLM if it is a new/open student or credit offer."""
    snaps = json.loads(PAGES.read_text(encoding="utf-8")) if PAGES.exists() else {}
    alerts = []
    for pg in cfg.get("watch_pages", []):
        try:
            html = requests.get(pg["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=20).text
            soup = BeautifulSoup(html, "html.parser")
            for t in soup(["script", "style", "noscript", "svg"]):
                t.decompose()
            lines = [l.strip() for l in soup.get_text("\n").splitlines() if len(l.strip()) > 25]
        except Exception as ex:
            print(f"[warn] page {pg['name']}: {ex}", file=sys.stderr)
            continue
        old = snaps.get(pg["url"])
        snaps[pg["url"]] = lines[:600]
        if old is None:
            continue  # first run: just record the baseline
        added = [l for l in difflib.unified_diff(old, lines[:600], lineterm="", n=0)
                 if l.startswith("+") and not l.startswith("+++")]
        if not added:
            continue
        ans = llm("You watch pages for NEW free credits / student offers / discounts / grants for AI tools. "
                   "Given added lines from a page, reply 'NO' if it is only noise or unrelated. Otherwise reply "
                   "with one or two sentences: what the offer is, who is eligible, and any deadline.",
                  f"Page: {pg['name']}\n" + "\n".join(added[:60]), 300).strip()
        if not ans.upper().startswith("NO"):
            alerts.append({"source": pg["name"], "title": f"Offer change: {pg['name']}", "link": pg["url"],
                           "score": 10, "summary": ans})
    PAGES.write_text(json.dumps(snaps), encoding="utf-8")
    return alerts


def notify_telegram(picks):
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat):
        print("[info] Telegram not configured; skipping push")
        return
    for p in picks[:10]:
        body = f"*{p['title']}*\n_{p['source']}_ · {p['score']}/10\n{p['summary']}\n{p['link']}"
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": body, "parse_mode": "Markdown"}, timeout=20)
        if r.status_code == 400 and "parse" in r.text:  # stray Markdown characters: resend as plain text
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              json={"chat_id": chat, "text": body}, timeout=20)
        if not r.ok:
            raise RuntimeError(f"Telegram {r.status_code}: {r.json().get('description')}")


def notify_email(picks):
    """One digest email per run. Gmail: use an App Password (needs 2-step verification on)."""
    user, pw, to = os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD"), os.getenv("EMAIL_TO", os.getenv("SMTP_USER"))
    if not (user and pw):
        print("[info] Email not configured; skipping")
        return
    msg = EmailMessage()
    msg["Subject"] = f"AI digest: {len(picks)} updates ({datetime.date.today()})"
    msg["From"], msg["To"] = user, to
    msg.set_content("\n\n".join(f"{p['title']}\n{p['source']} - {p['score']}/10\n{p['summary']}\n{p['link']}"
                                for p in picks))
    with smtplib.SMTP_SSL(os.getenv("SMTP_HOST", "smtp.gmail.com"), 465, timeout=30) as s:
        s.login(user, pw)
        s.send_message(msg)


def write_digest(picks):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"## {stamp}\n"] + [f"- **[{p['title']}]({p['link']})** ({p['source']}, {p['score']}/10) - {p['summary']}"
                                 for p in picks]
    old = DIGEST.read_text(encoding="utf-8") if DIGEST.exists() else "# AI Digest\n\n"
    head, _, rest = old.partition("\n\n")
    DIGEST.write_text(head + "\n\n" + "\n".join(lines) + "\n\n" + rest, encoding="utf-8")


def run_once():
    cfg = yaml.safe_load((HERE / "sources.yaml").read_text(encoding="utf-8"))
    seen = load_seen()
    items = fetch_new(cfg, seen)
    print(f"{len(items)} new items")
    alerts = check_pages(cfg)
    if items or alerts:
        picks = alerts + (rank(cfg["interests"], items) if items else [])
        print(f"{len(picks)} worth reading")
        if picks:
            write_digest(picks)
            for send in (notify_email, notify_telegram):
                try:
                    send(picks)
                except Exception as ex:
                    print(f"[warn] {send.__name__} failed: {ex}", file=sys.stderr)
        seen.update(i["link"] for i in items)
        SEEN.write_text(json.dumps(sorted(seen)[-5000:]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, metavar="MINUTES", help="run forever")
    ap.add_argument("--test", action="store_true", help="send one test message to email/Telegram and exit")
    a = ap.parse_args()
    if a.test:
        demo = [{"title": "Test message from your AI news agent", "source": "agent", "score": 10,
                 "summary": "If you can read this, delivery works.", "link": "https://example.com"}]
        for send in (notify_email, notify_telegram):
            try:
                send(demo)
                print(f"{send.__name__}: done")
            except Exception as ex:
                print(f"{send.__name__} failed: {ex}")
        sys.exit()
    while True:
        try:
            run_once()
        except Exception as ex:
            print(f"[error] {ex}", file=sys.stderr)
        if not a.loop:
            break
        time.sleep(a.loop * 60)
