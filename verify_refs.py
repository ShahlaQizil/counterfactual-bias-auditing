"""
Verify a BibTeX file (default: references.bib) against authoritative sources: the ACL Anthology's
official BibTeX for ACL papers, and Crossref (by DOI) for everything else.
Compares title, year, author surnames and pages, and prints each entry's DOI.

Run:
    python verify_refs.py [path/to/references.bib]
"""

from __future__ import annotations

import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path[:0] = [_here, _os.path.dirname(_here)]
import project  # noqa: E402  (finds the other code folders; runs from "Results and data")
project.setup()

import json
import re
import sys
import unicodedata
import urllib.request

# path to the BibTeX file: first command-line argument, default references.bib
BIB = sys.argv[1] if len(sys.argv) > 1 else _os.path.join(project.PAPER, "references.bib")

# key -> ("acl", anthology id) | ("doi", doi) | ("arxiv", id) | ("none", reason)
SOURCES = {
    "alnegheimish2022": ("acl", "2022.naacl-main.203"),
    "blodgett2020": ("acl", "2020.acl-main.485"),
    "blodgett2021": ("acl", "2021.acl-long.81"),
    "czarnowska2021": ("acl", "2021.tacl-1.74"),
    "delobelle2022": ("acl", "2022.naacl-main.122"),
    "dev2021": ("acl", "2021.emnlp-main.150"),
    "field2021": ("acl", "2021.acl-long.149"),
    "goldfarb2021": ("acl", "2021.acl-long.150"),
    "huang2020": ("acl", "2020.findings-emnlp.7"),
    "kiritchenko2018": ("acl", "S18-2005"),
    "maas2011": ("acl", "P11-1015"),
    "card2020": ("acl", "2020.emnlp-main.745"),
    "dror2018": ("acl", "P18-1128"),
    "smith2022": ("acl", "2022.emnlp-main.625"),
    "pleiss2017": ("none", "NeurIPS 2017 (no Crossref DOI); pages 5680-5689 per proceedings metadata"),
    "krishna2022": ("acl", "2022.acl-long.401"),
    "goldfarb2023": ("acl", "2023.findings-acl.139"),
    "gallegos2024": ("doi", "10.1162/coli_a_00524"),
    "orgad2022": ("acl", "2022.gebnlp-1.17"),
    "dev2022": ("acl", "2022.findings-aacl.24"),
    "kuzucu2024": ("doi", "10.1613/jair.1.16041"),
    "vanderwal2024": ("doi", "10.1613/jair.1.15195"),
    "mohammad2018": ("acl", "L18-1027"),
    "larson2017": ("acl", "W17-1601"),
    "prabhakaran2019": ("acl", "D19-1578"),
    "ribeiro2020": ("acl", "2020.acl-main.442"),
    "selvam2023": ("acl", "2023.acl-short.118"),
    "zhang2021": ("acl", "2021.naacl-main.305"),
    "bertrand2004": ("doi", "10.1257/0002828042002561"),
    "caliskan2017": ("doi", "10.1126/science.aal4230"),
    "dixon2018": ("doi", "10.1145/3278721.3278729"),
    "garg2019": ("doi", "10.1145/3306618.3317950"),
    "gaddis2017": ("doi", "10.15195/v4.a19"),
    "mitchell2019": ("doi", "10.1145/3287560.3287596"),
    "raji2020": ("doi", "10.1145/3351095.3372873"),
    "jacobs2021": ("doi", "10.1145/3442188.3445901"),
    "seshadri2022": ("arxiv", "2210.04337"),
    "guo2017": ("none", "PMLR v70 (no Crossref DOI); check https://proceedings.mlr.press/v70/guo17a.html"),
    "kusner2017": ("none", "NeurIPS 2017 (no Crossref DOI); check papers.nips.cc"),
}


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t)
    t = re.sub(r"[{}\\\"'`~^]|\\[a-z]+", "", t)
    return re.sub(r"[^a-z0-9]", "", t.lower())


def surname_set(authors: str) -> set[str]:
    out = set()
    for a in re.split(r"\s+and\s+", authors):
        a = a.strip()
        last = a.split(",")[0] if "," in a else a.split()[-1]
        out.add(norm(last))
    return out


def get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "verify_refs/1.0 (mailto:none)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def field(entry: str, name: str) -> str:
    m = re.search(rf"\b{name}\s*=\s*[{{\"](.*?)[}}\"]\s*,?\s*\n", entry, flags=re.S | re.I)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def local_entries() -> dict[str, str]:
    text = open(BIB, encoding="utf-8").read()
    return {m.group(1): m.group(0) for m in re.finditer(r"@\w+\{([^,]+),.*?\n\}", text, flags=re.S)}


def main() -> None:
    entries = local_entries()
    problems = 0
    for key, entry in sorted(entries.items()):
        kind, ref = SOURCES.get(key, ("none", "no source listed"))
        mine = {"title": field(entry, "title"), "year": field(entry, "year"),
                "author": field(entry, "author"), "pages": field(entry, "pages")}
        if kind == "none":
            print(f"[manual] {key}: {ref}")
            continue
        try:
            if kind == "acl":
                b = get(f"https://aclanthology.org/{ref}.bib")
                off = {"title": field(b, "title"), "year": field(b, "year"),
                       "author": field(b, "author"), "pages": field(b, "pages"), "doi": field(b, "doi")}
            elif kind == "doi":
                m = json.loads(get(f"https://api.crossref.org/works/{ref}"))["message"]
                sub = (m.get("subtitle") or [""])[0]  # Crossref stores subtitles separately
                off = {"title": (m.get("title") or [""])[0] + (": " + sub if sub else ""),
                       "year": str((m.get("issued", {}).get("date-parts") or [[""]])[0][0]),
                       "author": " and ".join(f"{a.get('family', '')}, {a.get('given', '')}" for a in m.get("author", [])),
                       "pages": m.get("page", ""), "doi": ref}
            else:  # arxiv
                a = get(f"http://export.arxiv.org/api/query?id_list={ref}")
                off = {"title": re.search(r"<entry>.*?<title>(.*?)</title>", a, re.S).group(1),
                       "year": re.search(r"<published>(\d{4})", a).group(1),
                       "author": " and ".join(re.findall(r"<name>(.*?)</name>", a)), "pages": "",
                       "doi": f"arXiv:{ref}"}
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {key}: could not fetch {kind}:{ref} ({e})")
            problems += 1
            continue
        issues = []
        if norm(mine["title"]) != norm(off["title"]):
            issues.append(f"title\n      bib: {mine['title']}\n      off: {off['title']}")
        if mine["year"] != off["year"]:
            issues.append(f"year bib={mine['year']} off={off['year']}")
        if surname_set(mine["author"]) != surname_set(off["author"]):
            issues.append(f"authors\n      bib: {mine['author']}\n      off: {off['author']}")
        if mine["pages"] and off["pages"] and norm(mine["pages"]) != norm(off["pages"]):
            issues.append(f"pages bib={mine['pages']} off={off['pages']}")
        tag = "OK     " if not issues else "MISMATCH"
        print(f"[{tag}] {key}  doi={off.get('doi', '')}  pages_official={off.get('pages', '')}")
        for i in issues:
            print("    -", i)
        problems += bool(issues)
    print(f"\n{problems} entries need attention")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
