#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estrae proverbi italiani da Wikiquote e li salva in CSV: idiom,gloss,tags.

- Legge i bullet dalla pagina "Proverbi_italiani"
- Se i bullet sono pochi, integra con i titoli dalla "Categoria:Proverbi_italiani"
- Per le voci che contengono almeno un link [[...]], scarica l'intro della pagina linkata come gloss
- Corregge il problema delle parentesi quadre: prima risolve i link doppi [[...]], poi rimuove solo [ ... ] (note)

Licenza: i contenuti provengono da Wikiquote (CC BY-SA 3.0).
Mantieni attribuzione se riusi/redistribuisci il CSV.
"""

import re
import csv
import time
import argparse
import requests
from typing import List, Dict

API = "https://it.wikiquote.org/w/api.php"
DEFAULT_PAGE = "Proverbi_italiani"
DEFAULT_CATEGORY = "Categoria:Proverbi_italiani"
DEFAULT_OUT = "italian_wikiquote_proverbs.csv"
UA = "CulturalStorytellingBot/1.0 (academic; contact: you@example.com)"


# -------------------- API helper --------------------
def mw_api(params: Dict) -> Dict:
    params = dict(params)
    params["format"] = "json"
    headers = {"User-Agent": UA}
    r = requests.get(API, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


# -------------------- text utils --------------------
def normalize_wikilinks_visible(text: str) -> str:
    """
    Trasforma [[A|B]] -> B, [[A]] -> A.
    Poi rimuove SOLO le parentesi quadre singole [ ... ] (note), NON i doppi [[...]].
    Elimina <ref>...</ref>, template {{...}}, entità e spazi extra.
    """
    if not text:
        return ""

    # 1) sostituisci i link wiki doppi [[...]] con il testo visibile
    def repl_link(m):
        inner = m.group(1)
        if "|" in inner:
            return inner.split("|", 1)[1]
        return inner
    text = re.sub(r"\[\[([^\]]+)\]\]", repl_link, text)

    # 2) rimuovi SOLO le parentesi quadre singole (note)
    text = re.sub(r"(?<!\[)\[[^\[\]]+\](?!\])", " ", text)

    # 3) rimuovi <ref>...</ref> e template {{...}}
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.DOTALL)
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)

    # 4) pulizia finale
    text = text.replace("“","").replace("”","").replace("«","").replace("»","")
    text = re.sub(r"&[^;\s]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .;:–—-[]")
    return text.strip()


def find_wikilinks(text: str) -> List[str]:
    """
    Estrae i TITOLI dei link wiki dalla forma [[...]].
    Da chiamare sul testo RAW (prima di normalizzarlo).
    """
    links = []
    for m in re.finditer(r"\[\[([^\]]+)\]\]", text):
        inner = m.group(1)
        title = inner.split("|", 1)[0].strip()
        if title:
            links.append(title)
    return links


# -------------------- extraction from PAGE --------------------
def extract_from_wikitext(page: str, min_tokens: int = 3) -> List[Dict]:
    """
    Usa action=parse&prop=wikitext per leggere le righe bullet (*, **, #).
    Ritorna una lista di dict: {"visible": testo_pulito, "links": [titoli_wiki]}
    """
    data = mw_api({"action": "parse", "page": page, "prop": "wikitext"})
    wt = data.get("parse", {}).get("wikitext", {}).get("*", "")
    if not wt:
        return []

    # Righe con bullet e numerate
    lines = re.findall(r"^[\*\#]+\s*(.+)$", wt, flags=re.MULTILINE)
    items = []
    for raw in lines:
        # non toccare il RAW prima di estrarre i link
        links = find_wikilinks(raw)
        visible = normalize_wikilinks_visible(raw)
        if not visible or len(visible.split()) < min_tokens:
            continue
        items.append({"visible": visible, "links": links})

    # de-dup per testo visibile
    seen, out = set(), []
    for it in items:
        k = it["visible"].lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


# -------------------- extraction from CATEGORY --------------------
def extract_titles_from_category(category: str, min_tokens: int = 2, limit: int = 500) -> List[Dict]:
    """
    Interroga la categoria e prende i titoli delle pagine (spesso il proverbio stesso).
    Ritorna [{"visible": titolo, "links": [titolo]}]
    """
    titles = []
    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": limit,
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = mw_api(params)
        cms = data.get("query", {}).get("categorymembers", [])
        for cm in cms:
            title = (cm.get("title") or "").strip()
            if title and not title.startswith("Categoria:"):
                titles.append(title)
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break

    # pulizia e filtro
    seen, items = set(), []
    for t in titles:
        if len(t.split()) < min_tokens:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        items.append({"visible": t, "links": [t]})
    return items


# -------------------- fetch gloss from linked pages --------------------
def fetch_intro_extracts(titles: List[str], sleep: float = 0.1) -> Dict[str, str]:
    """
    Usa action=query&prop=extracts&exintro&explaintext per ottenere l'intro dei titoli.
    Ritorna {title: intro_plain_text}
    """
    title_to_extract = {}
    if not titles:
        return title_to_extract

    BATCH = 20
    for i in range(0, len(titles), BATCH):
        chunk = titles[i:i+BATCH]
        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "titles": "|".join(chunk),
        }
        data = mw_api(params)
        pages = data.get("query", {}).get("pages", {})
        for _, page in pages.items():
            title = page.get("title", "")
            extract = (page.get("extract") or "").strip()
            if title:
                title_to_extract[title] = extract
        time.sleep(sleep)
    return title_to_extract


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="Estrai proverbi italiani da Wikiquote in CSV (idiom,gloss,tags).")
    ap.add_argument("--page", default=DEFAULT_PAGE, help="Pagina Wikiquote da cui leggere i bullet (default: Proverbi_italiani)")
    ap.add_argument("--category", default=DEFAULT_CATEGORY, help="Categoria di fallback (default: Categoria:Proverbi_italiani)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="File CSV di output")
    ap.add_argument("--min_tokens", type=int, default=3, help="Min parole per tenere una voce dai bullet")
    ap.add_argument("--follow_links", action="store_true", help="Scarica intro delle pagine linkate per usarle come gloss")
    ap.add_argument("--max_gloss_chars", type=int, default=300, help="Max caratteri per la gloss")
    args = ap.parse_args()

    print(f"Fetching wikitext from page: {args.page}")
    items_page = extract_from_wikitext(args.page, min_tokens=args.min_tokens)
    print(f"From page bullets: {len(items_page)} items")

    items = items_page
    if len(items_page) < 50:
        print(f"Also querying category: {args.category}")
        items_cat = extract_titles_from_category(args.category, min_tokens=2)
        print(f"From category titles: {len(items_cat)} items")
        # merge deduplicando per 'visible'
        seen, merged = set(), []
        for it in items_page + items_cat:
            k = it["visible"].lower()
            if k in seen:
                continue
            seen.add(k)
            merged.append(it)
        items = merged

    # opzionale: scarica le intro per le gloss
    link_targets = []
    if args.follow_links:
        for it in items:
            if it["links"]:
                link_targets.append(it["links"][0])
        link_targets = sorted(set(link_targets))
        print(f"Link targets to fetch (for gloss): {len(link_targets)}")
        extracts = fetch_intro_extracts(link_targets)
    else:
        extracts = {}

    # costruisci righe CSV
    rows = []
    for it in items:
        idiom = it["visible"]
        gloss = ""
        if args.follow_links and it["links"]:
            t = it["links"][0]
            gloss = (extracts.get(t) or "").strip().split("\n")[0]
            if len(gloss) > args.max_gloss_chars:
                gloss = gloss[: args.max_gloss_chars - 3].rstrip() + "..."
        rows.append((idiom, gloss, "proverb,wikiquote,it"))

    # dedup finale
    seen, final_rows = set(), []
    for idiom, gloss, tags in rows:
        k = idiom.lower()
        if k in seen:
            continue
        seen.add(k)
        final_rows.append((idiom, gloss, tags))

    print(f"Total distinct proverbs exported: {len(final_rows)}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idiom", "gloss", "tags"])
        for r in final_rows:
            w.writerow(r)

    print(f"✅ Wrote {args.out}")


if __name__ == "__main__":
    main()
