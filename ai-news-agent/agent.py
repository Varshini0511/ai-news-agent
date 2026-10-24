"""AI news agent: fetch feeds -> an LLM (Gemini/Grok/Groq/Claude) scores/summarizes -> push to Telegram + digest.md.

Run once:      python agent.py
Run forever:   python agent.py --loop 30     (every 30 minutes)
"""
import argparse, json, os, sys, time, datetime, pathlib
import calendar, difflib, smtplib
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
MIN_SCORE = int(os.getenv("MIN_SCORE", "6"))
BREAKING_SCORE = int(os.getenv("BREAKING_SCORE", "9"))  # sent immediately, individually, with a siren
MAX_AGE_HOURS = int(os.getenv("MAX_AGE_HOURS", "72"))  # older articles are skipped, even if never seen
PER_SOURCE = int(os.getenv("PER_SOURCE", "3"))  # at most this many picks per source per run, so one site can't flood the feed
PAGE_CHECK_HOURS = 24  # watched offer pages are checked once a day, not every run
PAGE_COOLDOWN_DAYS = 14  # and a page alerts at most once per 14 days
OFFER_WORDS = ("credit", "free", "student", "offer", "discount", "grant", "promo", "trial", "$", "%")


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


def _age_hours(e):
    t = e.get("published_parsed") or e.get("updated_parsed")
    return (time.time() - calendar.timegm(t)) / 3600 if t else None  # feedparser times are UTC


def fetch_new(cfg, seen):
    """New, recent items only. Old ones are returned separately so they get marked seen and never come back."""
    items, stale = [], []
    for f in cfg["feeds"]:
        try:
            for e in feedparser.parse(f["url"], agent="Mozilla/5.0 ai-news-agent").entries[:15]:
                link = e.get("link")
                if not link or link in seen:
                    continue
                age = _age_hours(e)
                if age is not None and age > MAX_AGE_HOURS:
                    stale.append(link)
                    continue
                text = BeautifulSoup(e.get("summary") or "", "html.parser").get_text(" ", strip=True)
                items.append({"source": f["name"], "title": e.get("title", ""), "link": link, "text": text[:600]})
        except Exception as ex:
            print(f"[warn] {f['name']}: {ex}", file=sys.stderr)
    return items, stale


def rank(interests, items):
    """One LLM call per batch of 15. Returns (picks, links that were actually scored).
    Items in a failed batch (quota, bad output) are not marked scored, so they are retried next run."""
    out, done = [], []
    for i in range(0, len(items), 15):
        batch = items[i:i + 15]
        listing = "\n".join(f"{n}. [{it['source']}] {it['title']} - {it['text'][:280]}"
                            for n, it in enumerate(batch))
        try:
            text = llm("You curate a daily AI news feed for a working software engineer who is learning AI. "
                       "Scores 9-10 are RARE: a major new model or product release, a big API or pricing change. "
                       "Useful releases, tools and tutorials are 6-8. Memes, questions, complaints, rumours, gossip, "
                       "funding news and vague posts are 1-4. "
                       "For each item write a summary of 2 short plain-English sentences: what actually happened, then "
                       "why it is useful to someone building with AI. No jargon or hype; explain any acronym. "
                       'Reply with ONLY a JSON list: [{"id":int,"score":1-10,"summary":"..."}] for every item.',
                       f"Interests: {interests}\n\nItems:\n{listing}", 3000)
            rows = json.loads(text[text.index("["): text.rindex("]") + 1])
        except Exception as ex:
            print(f"[warn] ranking batch skipped, will retry next run: {ex}", file=sys.stderr)
            continue
        done += [it["link"] for it in batch]
        for r in rows:
            if isinstance(r.get("id"), int) and 0 <= r["id"] < len(batch) and r.get("score", 0) >= MIN_SCORE:
                out.append({**batch[r["id"]], **r})
    out.sort(key=lambda x: -x["score"])
    per, picks = {}, []
    for p in out:  # keep the feed varied
        per[p["source"]] = per.get(p["source"], 0) + 1
        if per[p["source"]] <= PER_SOURCE:
            picks.append(p)
    return picks, done


def check_pages(cfg):
    """Snapshot watched pages; when text changes, ask the LLM if it is a new/open student or credit offer."""
    snaps = json.loads(PAGES.read_text(encoding="utf-8")) if PAGES.exists() else {}
    meta = snaps.setdefault("_meta", {})
    now = time.time()
    if now - meta.get("checked", 0) < PAGE_CHECK_HOURS * 3600:
        return []
    meta["checked"] = now
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
        added = [l for l in added if any(w in l.lower() for w in OFFER_WORDS)]
        if not added or now - meta.get("alerted", {}).get(pg["url"], 0) < PAGE_COOLDOWN_DAYS * 86400:
            continue
        try:
            ans = llm("You watch pages for NEW free credits / student offers / discounts / grants for AI tools. "
                      "You get only the lines ADDED to the page since yesterday. Most changes are noise: reworded "
                      "text, rotating banners, dates, an offer that already existed. Reply exactly NO unless the "
                      "added lines clearly announce a NEW offer. If they do, reply with two plain sentences: what "
                      "the new offer is, and who can get it (plus any deadline).",
                      f"Page: {pg['name']}\n" + "\n".join(added[:60]), 800).strip()
        except Exception as ex:
            print(f"[warn] page {pg['name']}: {ex}", file=sys.stderr)
            continue
        if ans and not ans.upper().startswith("NO"):
            meta.setdefault("alerted", {})[pg["url"]] = now
            alerts.append({"source": pg["name"], "title": f"New offer: {pg['name']}", "link": pg["url"],
                           "score": 8, "summary": ans})
    PAGES.write_text(json.dumps(snaps), encoding="utf-8")
    return alerts


def _tg_send(token, chat, body, markdown=True):
    payload = {"chat_id": chat, "text": body, "disable_web_page_preview": not markdown}
    if markdown:
        payload["parse_mode"] = "Markdown"
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json=payload, timeout=20)
    if r.status_code == 400 and markdown:  # stray Markdown characters: resend as plain text
        return _tg_send(token, chat, body, markdown=False)
    if not r.ok:
        raise RuntimeError(f"Telegram {r.status_code}: {r.json().get('description')}")


def notify_telegram(picks):
    """Breaking items (score >= BREAKING_SCORE) go out one by one with a siren; the rest as one bundled digest."""
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat):
        print("[info] Telegram not configured; skipping push")
        return
    breaking = [p for p in picks if p["score"] >= BREAKING_SCORE]
    rest = [p for p in picks if p["score"] < BREAKING_SCORE]
    for p in breaking[:8]:
        _tg_send(token, chat, f"🚨 *BREAKING* ({p['score']}/10)\n*{p['title']}*\n_{p['source']}_\n{p['summary']}\n{p['link']}")
    if rest:
        lines = [f"📰 *AI digest* - {len(rest)} updates"] + [
            f"\n*{p['title']}* ({p['source']}, {p['score']}/10)\n{p['summary']}\n{p['link']}" for p in rest[:12]]
        _tg_send(token, chat, "\n".join(lines)[:4000])


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
    items, stale = fetch_new(cfg, seen)
    print(f"{len(items)} new items ({len(stale)} older than {MAX_AGE_HOURS}h skipped)")
    alerts = check_pages(cfg)
    ranked, done = rank(cfg["interests"], items) if items else ([], [])
    picks = alerts + ranked
    print(f"{len(picks)} worth reading; {len(items) - len(done)} left for the next run")
    if picks:
        write_digest(picks)
        for send in (notify_email, notify_telegram):
            try:
                send(picks)
            except Exception as ex:
                print(f"[warn] {send.__name__} failed: {ex}", file=sys.stderr)
    seen.update(stale + done)
    SEEN.write_text(json.dumps(sorted(seen)[-8000:]))

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
