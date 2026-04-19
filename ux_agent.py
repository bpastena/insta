#!/usr/bin/env python3
"""UX Analysis Agent - Analyzes user experience of web pages using Claude."""

import json
import re
import sys
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
import anthropic
from anthropic import beta_tool

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a senior UX analyst with expertise in web usability, accessibility \
(WCAG 2.1), and conversion optimization. Analyze web pages thoroughly and deliver structured, \
actionable reports grounded in specific page elements you discover.

Analysis framework:
• Navigation & IA: menu structure, breadcrumbs, search, wayfinding clarity
• Content Hierarchy: heading structure, visual hierarchy, readability, whitespace
• CTAs: prominence, clarity, placement, friction reduction, urgency signals
• Mobile-Friendliness: viewport meta, responsive indicators, touch-target sizes
• Accessibility: WCAG hints, ARIA usage, keyboard navigation, color contrast cues
• Performance UX: lazy loading, blocking resources, skeleton screens, resource hints

Be specific—reference actual text, elements, and patterns found on the page."""


# ─── Tools ────────────────────────────────────────────────────────────────────

@beta_tool
def fetch_page(url: str) -> str:
    """Fetch the HTML content of a web page for UX analysis.

    Args:
        url: The full URL to fetch, including the scheme (https://...).
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        with httpx.Client(follow_redirects=True, timeout=30) as http:
            resp = http.get(url, headers=headers)
            resp.raise_for_status()
        return json.dumps({
            "status_code": resp.status_code,
            "final_url": str(resp.url),
            "content_type": resp.headers.get("content-type", ""),
            # Cap at 60 KB to stay within model context limits
            "html": resp.text[:60_000],
            "total_bytes": len(resp.content),
        })
    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"HTTP {e.response.status_code}: {e.response.reason_phrase}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@beta_tool
def analyze_structure(html: str) -> str:
    """Analyze structural elements of a page: headings, navigation, landmarks, and metadata.

    Args:
        html: Raw HTML content of the page (as returned by fetch_page).
    """
    soup = BeautifulSoup(html, "html.parser")

    # Heading hierarchy
    headings = [
        {"level": h.name, "text": h.get_text(strip=True)[:200]}
        for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
    ]

    # Navigation elements
    nav_items = []
    for nav in soup.find_all("nav"):
        links = [
            {"text": a.get_text(strip=True)[:80], "href": a.get("href", "")}
            for a in nav.find_all("a")[:20]
        ]
        nav_items.append({"aria_label": nav.get("aria-label", ""), "links": links})
    for elem in soup.find_all(attrs={"role": "navigation"}):
        if elem.name != "nav":
            links = [
                {"text": a.get_text(strip=True)[:80], "href": a.get("href", "")}
                for a in elem.find_all("a")[:20]
            ]
            nav_items.append({"role_navigation": True, "links": links})

    # Semantic landmark counts
    landmarks = {
        tag: len(soup.find_all(tag))
        for tag in ["header", "main", "footer", "aside", "article", "section", "nav"]
    }
    for role in ["banner", "main", "contentinfo", "search", "complementary"]:
        count = len(soup.find_all(attrs={"role": role}))
        if count:
            landmarks[f"role={role}"] = count

    # Page metadata
    html_tag = soup.find("html")
    meta: dict = {
        "title": soup.title.string.strip() if soup.title and soup.title.string else "",
        "description": "",
        "viewport": "",
        "language": html_tag.get("lang", "") if html_tag else "",
    }
    for m in soup.find_all("meta"):
        name = m.get("name", "").lower()
        if name == "description":
            meta["description"] = m.get("content", "")[:300]
        elif name == "viewport":
            meta["viewport"] = m.get("content", "")

    h1_texts = [h["text"] for h in headings if h["level"] == "h1"]

    return json.dumps({
        "headings": headings,
        "h1_count": len(h1_texts),
        "h1_texts": h1_texts,
        "navigation": nav_items,
        "landmarks": landmarks,
        "meta": meta,
    }, ensure_ascii=False)


@beta_tool
def analyze_interactive(html: str) -> str:
    """Analyze interactive elements: buttons/CTAs, forms, and links.

    Args:
        html: Raw HTML content of the page (as returned by fetch_page).
    """
    soup = BeautifulSoup(html, "html.parser")

    # Buttons and CTAs
    buttons = []
    for btn in soup.find_all("button"):
        buttons.append({
            "text": btn.get_text(strip=True)[:100],
            "type": btn.get("type", "button"),
            "aria_label": btn.get("aria-label", ""),
        })
    for inp in soup.find_all("input", type=["submit", "button", "reset"]):
        buttons.append({
            "text": inp.get("value", inp.get("type", "")),
            "type": inp.get("type", ""),
        })
    for a in soup.find_all("a"):
        cls = " ".join(a.get("class", []))
        if re.search(r"\b(btn|button|cta)\b", cls, re.I):
            buttons.append({
                "text": a.get_text(strip=True)[:100],
                "href": a.get("href", ""),
                "type": "link-button",
                "classes": cls[:80],
            })

    # Forms
    forms = []
    for form in soup.find_all("form"):
        fields = []
        for field in form.find_all(["input", "textarea", "select"]):
            ftype = field.get("type", field.name)
            if ftype == "hidden":
                continue
            label_text = ""
            fid = field.get("id")
            if fid:
                lbl = soup.find("label", attrs={"for": fid})
                if lbl:
                    label_text = lbl.get_text(strip=True)
            if not label_text:
                label_text = field.get("aria-label", "") or field.get("placeholder", "")
            fields.append({
                "type": ftype,
                "label": label_text,
                "required": field.has_attr("required"),
            })
        forms.append({
            "method": form.get("method", "GET").upper(),
            "action": form.get("action", ""),
            "field_count": len(fields),
            "fields": fields[:15],
        })

    # Link analysis
    all_links = soup.find_all("a", href=True)
    new_tab_links = [a for a in all_links if a.get("target") == "_blank"]
    empty_text_links = [
        a for a in all_links
        if not a.get_text(strip=True) and not a.get("aria-label")
    ]

    return json.dumps({
        "buttons": buttons[:30],
        "button_count": len(buttons),
        "forms": forms,
        "links": {
            "total": len(all_links),
            "external": len([a for a in all_links if a.get("href", "").startswith("http")]),
            "open_new_tab": len(new_tab_links),
            "empty_text_count": len(empty_text_links),
            "new_tab_examples": [a.get_text(strip=True)[:60] for a in new_tab_links[:5]],
        },
    }, ensure_ascii=False)


@beta_tool
def check_accessibility(html: str) -> str:
    """Audit HTML for accessibility compliance indicators (WCAG 2.1 hints).

    Args:
        html: Raw HTML content of the page (as returned by fetch_page).
    """
    soup = BeautifulSoup(html, "html.parser")
    issues = []
    passes = []

    # lang attribute on <html>
    html_el = soup.find("html")
    if html_el and html_el.get("lang"):
        passes.append(f"Language declared: lang='{html_el.get('lang')}'")
    else:
        issues.append("Missing lang attribute on <html> element")

    # Image alt text
    imgs = soup.find_all("img")
    no_alt = [img.get("src", "")[:60] for img in imgs if not img.has_attr("alt")]
    if no_alt:
        issues.append(f"{len(no_alt)} image(s) missing alt text: {no_alt[:3]}")
    elif imgs:
        passes.append(f"All {len(imgs)} images have alt attributes")

    # Form input labels
    inputs = [
        i for i in soup.find_all("input")
        if i.get("type") not in ("hidden", "submit", "button", "reset", "image")
    ]
    unlabeled = []
    for inp in inputs:
        fid = inp.get("id")
        labeled = (
            (fid and soup.find("label", attrs={"for": fid}))
            or inp.get("aria-label")
            or inp.get("aria-labelledby")
            or inp.get("title")
        )
        if not labeled:
            unlabeled.append(inp.get("name") or inp.get("type") or "?")
    if unlabeled:
        issues.append(f"Form inputs without accessible labels: {unlabeled[:5]}")
    elif inputs:
        passes.append(f"All {len(inputs)} form inputs appear labeled")

    # Heading hierarchy
    headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
    h1s = [h for h in headings if h.name == "h1"]
    if len(h1s) == 1:
        passes.append("Single H1 heading (correct)")
    elif len(h1s) == 0:
        issues.append("No H1 heading found")
    else:
        issues.append(f"Multiple H1 headings ({len(h1s)}) — should be exactly one")

    # Skip navigation link
    first_links = soup.find_all("a", limit=5)
    has_skip = any(
        "skip" in (a.get_text(strip=True) + a.get("href", "")).lower()
        for a in first_links
    )
    if has_skip:
        passes.append("Skip-to-content link present")
    else:
        issues.append("No skip-to-content link detected (keyboard users benefit from this)")

    # Positive tabindex (disrupts natural order)
    positive_tabindex = [
        e.name for e in soup.find_all(attrs={"tabindex": True})
        if str(e.get("tabindex", "0")).lstrip("-").isdigit()
        and int(e.get("tabindex", 0)) > 0
    ]
    if positive_tabindex:
        issues.append(f"Positive tabindex disrupts tab order: {positive_tabindex[:5]}")

    # ARIA usage
    aria_labeled = soup.find_all(attrs={"aria-label": True})
    if aria_labeled:
        passes.append(f"ARIA labels on {len(aria_labeled)} element(s)")

    return json.dumps({
        "issues": issues,
        "passes": passes,
        "summary": {
            "total_images": len(imgs),
            "images_missing_alt": len(no_alt),
            "total_inputs": len(inputs),
            "unlabeled_inputs": len(unlabeled),
            "heading_count": len(headings),
            "aria_labeled_elements": len(aria_labeled),
        },
    }, ensure_ascii=False)


@beta_tool
def check_performance(html: str) -> str:
    """Analyze performance-related UX indicators: lazy loading, script loading, resource hints.

    Args:
        html: Raw HTML content of the page (as returned by fetch_page).
    """
    soup = BeautifulSoup(html, "html.parser")
    head = soup.find("head") or BeautifulSoup("", "html.parser")

    # Image loading
    imgs = soup.find_all("img")
    lazy_imgs = [i for i in imgs if i.get("loading") == "lazy"]
    imgs_with_dims = [i for i in imgs if i.get("width") and i.get("height")]

    # Script loading patterns
    external_scripts = soup.find_all("script", src=True)
    blocking = [s.get("src", "")[:80] for s in external_scripts
                if not s.has_attr("async") and not s.has_attr("defer")]
    async_count = sum(1 for s in external_scripts if s.has_attr("async"))
    deferred_count = sum(1 for s in external_scripts if s.has_attr("defer"))

    # Resource hints
    preloads = soup.find_all("link", rel="preload")
    prefetches = soup.find_all("link", rel="prefetch")
    dns_prefetch = soup.find_all("link", rel="dns-prefetch")
    preconnects = soup.find_all("link", rel="preconnect")

    # Viewport for mobile
    viewport_meta = soup.find("meta", attrs={"name": "viewport"})

    # Inline styles count (excessive = potential render-blocking)
    inline_style_tags = len(soup.find_all("style"))

    return json.dumps({
        "images": {
            "total": len(imgs),
            "lazy_loaded": len(lazy_imgs),
            "with_explicit_dimensions": len(imgs_with_dims),
            "lazy_coverage": f"{round(len(lazy_imgs) / len(imgs) * 100)}%" if imgs else "N/A",
            "note": "Explicit dimensions prevent Cumulative Layout Shift (CLS)",
        },
        "scripts": {
            "total_external": len(external_scripts),
            "render_blocking": blocking[:5],
            "async": async_count,
            "deferred": deferred_count,
        },
        "stylesheets": {"total": len(soup.find_all("link", rel="stylesheet"))},
        "inline_style_blocks": inline_style_tags,
        "resource_hints": {
            "preload": len(preloads),
            "prefetch": len(prefetches),
            "dns_prefetch": len(dns_prefetch),
            "preconnect": len(preconnects),
        },
        "mobile": {
            "viewport_meta": (
                viewport_meta.get("content", "") if viewport_meta else "MISSING — critical for mobile UX"
            ),
        },
    }, ensure_ascii=False)


# ─── Agent runner ──────────────────────────────────────────────────────────────

def analyze_url(url: str) -> tuple[str, list[str]]:
    """Run the UX analysis agent and return (report_text, tools_used)."""
    runner = client.beta.messages.tool_runner(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},   # cache stable system prompt
        }],
        tools=[
            fetch_page,
            analyze_structure,
            analyze_interactive,
            check_accessibility,
            check_performance,
        ],
        messages=[{
            "role": "user",
            "content": (
                f"Analyze the UX of this URL: {url}\n\n"
                "Step 1 — fetch the page.\n"
                "Step 2 — extract structural data (headings, navigation, landmarks, meta).\n"
                "Step 3 — extract interactive elements (buttons/CTAs, forms, links).\n"
                "Step 4 — run the accessibility audit.\n"
                "Step 5 — run the performance hints check.\n"
                "Step 6 — produce a comprehensive UX report with these sections:\n\n"
                "1. **Overall UX Score** (1–10) with a one-sentence rationale\n"
                "2. **Navigation & Information Architecture**\n"
                "3. **Content Hierarchy & Readability**\n"
                "4. **Calls-to-Action (CTAs)**\n"
                "5. **Mobile-Friendliness**\n"
                "6. **Accessibility**\n"
                "7. **Performance UX Hints**\n"
                "8. **Top 5 Priority Improvements** (ranked by user impact)\n"
                "9. **Quick Wins** (easy changes, high impact)\n\n"
                "Cite specific elements, text, or numbers from the page in each section."
            ),
        }],
    )

    report_text = ""
    tools_used: list[str] = []

    for message in runner:
        for block in message.content:
            if block.type == "tool_use":
                tools_used.append(block.name)
                print(f"  → {block.name}()", flush=True)
            elif block.type == "text" and block.text.strip():
                report_text = block.text   # final non-empty text block is the report

    return report_text, tools_used


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:   python ux_agent.py <URL>")
        print("Example: python ux_agent.py https://example.com")
        sys.exit(1)

    url = sys.argv[1]
    if not urlparse(url).scheme:
        url = "https://" + url

    print(f"\n📋  UX Analysis Agent")
    print(f"    Analyzing: {url}\n")
    print("Tools called:")

    report, tools = analyze_url(url)

    print("\n" + "=" * 70)
    print("  UX ANALYSIS REPORT")
    print("=" * 70)
    print(f"  URL   : {url}")
    print(f"  Tools : {', '.join(tools)}")
    print("-" * 70 + "\n")
    print(report)
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
