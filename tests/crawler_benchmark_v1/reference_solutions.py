from __future__ import annotations


AURORA_SOLUTION = r'''from __future__ import annotations
import json, sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen

class Reader(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows=[]; self.row=None; self.field=None; self.depth=0
    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs); classes=set(attrs.get("class", "").split())
        if tag == "article" and "velin-shell" in classes and attrs.get("data-glyph"):
            self.row={"sigil":attrs["data-glyph"],"label":"","group":"","rank":0,"detail_url":""}; self.depth=1
        elif self.row is not None:
            self.depth += 1
            if tag == "h3" and "morrow-ink" in classes: self.field="label"
            elif tag == "span" and "halo-band" in classes: self.field="group"
            elif tag == "i" and "index-flare" in classes: self.field="rank"
            elif tag == "a" and "path-knot" in classes: self.row["detail_url"]=attrs.get("href", "")
    def handle_data(self, data):
        if self.row is not None and self.field and data.strip():
            if self.field == "rank": self.row[self.field]=int(data.strip())
            else: self.row[self.field] += data.strip()
    def handle_endtag(self, tag):
        if self.row is None: return
        if tag in {"h3","span","i"}: self.field=None
        self.depth -= 1
        if self.depth == 0:
            self.rows.append(self.row); self.row=None

def collect(url):
    with urlopen(url, timeout=15) as response: text=response.read().decode()
    reader=Reader(); reader.feed(text)
    for row in reader.rows: row["detail_url"]=urljoin(url, row["detail_url"])
    return reader.rows

if __name__ == "__main__":
    Path(sys.argv[2]).write_text(json.dumps(collect(sys.argv[1]), indent=2), encoding="utf-8")
'''


KILN_SOLUTION = r'''from __future__ import annotations
import json, sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen

class Reader(HTMLParser):
    def __init__(self):
        super().__init__(); self.in_table=False; self.row=None; self.cell=None; self.depth=0; self.rows=[]
    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs)
        if tag == "table" and attrs.get("id") == "lithic-grid": self.in_table=True
        elif self.in_table and tag == "tr" and attrs.get("data-entry") == "yes": self.row={}
        elif self.row is not None and tag == "td" and attrs.get("data-col"):
            self.cell=attrs["data-col"]; self.row[self.cell]=""; self.depth=1
        elif self.cell is not None: self.depth += 1
    def handle_data(self, data):
        if self.row is not None and self.cell is not None: self.row[self.cell] += data
    def handle_endtag(self, tag):
        if self.cell is not None:
            self.depth -= 1
            if self.depth == 0: self.cell=None
        if tag == "tr" and self.row is not None:
            row={key:" ".join(value.split()) for key,value in self.row.items()}
            reading=row.get("reading", "").replace(",", "")
            self.rows.append({"sector":row["sector"],"station":row["station"],"reading":int(reading) if reading else None,"observed_at":row["observed_at"]}); self.row=None
        elif tag == "table" and self.in_table: self.in_table=False

def collect(url):
    with urlopen(url, timeout=15) as response: text=response.read().decode()
    reader=Reader(); reader.feed(text); return reader.rows

if __name__ == "__main__":
    Path(sys.argv[2]).write_text(json.dumps(collect(sys.argv[1]), indent=2), encoding="utf-8")
'''


TIDE_SOLUTION = r'''from __future__ import annotations
import json, sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen

class Reader(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows=[]; self.row=None; self.field=None; self.next_url=None
    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs); classes=set(attrs.get("class", "").split())
        if tag == "section" and "folio-slip" in classes: self.row={"mark":attrs["data-mark"],"caption":"","amount":0.0,"source_url":""}
        elif self.row is not None and tag == "b" and "folio-caption" in classes: self.field="caption"
        elif self.row is not None and tag == "span" and "folio-amount" in classes: self.field="amount"
        elif self.row is not None and tag == "a" and "folio-source" in classes: self.row["source_url"]=attrs.get("href", "")
        elif tag == "a" and attrs.get("rel") == "next": self.next_url=attrs.get("href")
    def handle_data(self, data):
        if self.row is not None and self.field and data.strip():
            self.row[self.field]=float(data.strip()) if self.field == "amount" else self.row[self.field]+data.strip()
    def handle_endtag(self, tag):
        if tag in {"b","span"}: self.field=None
        elif tag == "section" and self.row is not None: self.rows.append(self.row); self.row=None

def collect(start):
    url=start; seen=set(); result=[]
    while url:
        with urlopen(url, timeout=15) as response: text=response.read().decode()
        reader=Reader(); reader.feed(text)
        for row in reader.rows:
            if row["mark"] in seen: continue
            seen.add(row["mark"]); row["source_url"]=urljoin(url, row["source_url"]); result.append(row)
        url=urljoin(url, reader.next_url) if reader.next_url else None
    return result

if __name__ == "__main__":
    Path(sys.argv[2]).write_text(json.dumps(collect(sys.argv[1]), indent=2), encoding="utf-8")
'''


PRISM_SOLUTION = r'''from __future__ import annotations
import json, sys
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen
from xml.etree import ElementTree as ET

def local(tag): return tag.rsplit("}", 1)[-1]
def child_text(node, names):
    for child in node:
        if local(child.tag) in names: return " ".join("".join(child.itertext()).split())
    return None
def collect(url):
    with urlopen(url, timeout=20) as response: root=ET.fromstring(response.read())
    rows=[]
    for node in root.iter():
        if local(node.tag) not in {"item","entry"}: continue
        link=child_text(node, {"link"}) or ""
        for child in node:
            if local(child.tag) == "link" and child.get("href"): link=child.get("href", "")
        rows.append({"headline":child_text(node,{"title"}) or "","link":urljoin(url,link),"published":child_text(node,{"pubDate","published","updated"}) or "","source":child_text(node,{"creator","author","source"}) or "CNEOS","summary":child_text(node,{"description","summary","content"})})
    return rows
if __name__ == "__main__":
    Path(sys.argv[2]).write_text(json.dumps(collect(sys.argv[1]), indent=2), encoding="utf-8")
'''


ORBIT_SOLUTION = r'''from __future__ import annotations
import json, sys
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import urlopen

def page_url(source, cursor):
    parts=urlsplit(source); query=dict(parse_qsl(parts.query, keep_blank_values=True)); query["cursor"]=cursor
    return urlunsplit((parts.scheme,parts.netloc,parts.path,urlencode(query),parts.fragment))
def collect(source):
    cursor=""; seen=set(); result=[]
    while True:
        url=page_url(source,cursor)
        with urlopen(url, timeout=15) as response: payload=json.load(response)
        for row in payload["samples"]:
            if row["identity"] in seen: continue
            seen.add(row["identity"]); result.append({"identity":row["identity"],"title":row["title"],"measure":int(row["measure"]),"origin":urljoin(url,row["origin"])})
        cursor=payload.get("next_cursor")
        if not cursor: break
    return result
if __name__ == "__main__":
    Path(sys.argv[2]).write_text(json.dumps(collect(sys.argv[1]), indent=2), encoding="utf-8")
'''


SOLUTION_BY_CASE = {
    "opal_canopy": AURORA_SOLUTION,
    "flint_harbor": KILN_SOLUTION,
    "marble_tide": TIDE_SOLUTION,
    "violet_prism": PRISM_SOLUTION,
    "copper_orbit": ORBIT_SOLUTION,
    "cedar_signal": PRISM_SOLUTION,
}
