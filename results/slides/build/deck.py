"""Build results/slides/project2.key from project2_draft.md via AppleScript (Keynote, Basic White, 1920x1080)."""
import re, subprocess, sys
from PIL import Image

ROOT = "/Users/neel.shah/work/nlp1-2026"
MD = open(f"{ROOT}/results/slides/project2_draft.md").read()
FIG = f"{ROOT}/results/slides/figures"
OUT = f"{ROOT}/results/slides/project2.key"

def clean(t):
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t); t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", t)
    return t.replace("`", "").strip()

def tbd(t):  # visible placeholders: no project-management codes
    return "TBD" if "[ ]" in t else t

def q(t):  # AppleScript string literal
    t = t.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + t.replace("\n", '" & linefeed & "') + '"'

def parse(block):
    title = block.split("\n", 1)[0].split(": ", 1)[1].strip()
    src = block.split("**On slide (short)**")[1] if "**On slide (short)**" in block else block.split("**Bullets**")[1]
    bullets = [tbd(clean(re.sub(r"^\d+\.\s*", "", l))) for l in src.split("\n\n")[0].strip("\n").splitlines() if re.match(r"^\d+\.", l)]
    tables, cur = [], None
    for l in block.splitlines():
        if l.startswith("|"):
            if re.match(r"^\|[-| ]+\|$", l): continue
            cells = [tbd(clean(c)) for c in l.strip().strip("|").split("|")]
            (cur := cur if cur is not None else []).append(cells)
        elif cur is not None:
            tables.append(cur); cur = None
    if cur: tables.append(cur)
    caps = [tbd(clean(m)) for m in re.findall(r"\*Caption:\*\s*(.+)", block)]
    i = min([block.find(k) for k in ("**Sources:**", "**Notes**", "**Pipeline detail") if k in block] or [len(block)])
    notes = "BULLETS (full text)\n" + "\n".join(f"{n+1}. {b}" for n, b in enumerate(bullets)) + "\n\n" + clean(block[i:].replace("---", ""))
    return dict(title=title, bullets=bullets, tables=tables, caps=caps, notes=notes)

slides = [parse(b) for b in MD.split("\n## Slide ")[1:]]

L = ['tell application "Keynote"', "activate",
     'set d to make new document with properties {document theme:theme "Basic White", width:1920, height:1080}',
     "tell d"]
L += ['set s to slide 1', 'set base slide of s to master slide "Title"',
      'set object text of default title item of s to "Sentence encoders for SNLI"',
      'set object text of default body item of s to "Project 2  ·  Dario, Emil, Neel  ·  NLP1 2026"']

X0, XR, WR = 80, 1000, 840
def text_item(x, y, w, txt, size=15, grey=True):
    out = [f"set tb to make new text item with properties {{object text:{q(txt)}}}",
           f"set size of object text of tb to {size}", f"set width of tb to {w}", f"set position of tb to {{{x}, {y}}}"]
    if grey: out.append("set color of object text of tb to {26000, 26000, 26000}")
    return out
def table(x, y, w, rows, rh=42, fs=15, colw=None):
    r, c = len(rows), max(len(r) for r in rows)
    out = [f"set t to make new table with properties {{row count:{r}, column count:{c}, header row count:1, header column count:0, footer row count:0}}",
           f"set width of t to {w}", f"set height of t to {r*rh}", f"set position of t to {{{x}, {y}}}", "tell t"]
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            out.append(f"set value of cell {j+1} of row {i+1} to {q(v)}")
    out.append(f"set font size of cell range to {fs}")
    for j, cw in enumerate(colw or []):
        out.append(f"set width of column {j+1} to {cw}")
    out.append("end tell")
    return out, r * rh
def image(x, y, path, h):
    W, H = Image.open(path).size; w = int(h * W / H)
    return [f'make new image with properties {{file:POSIX file "{path}", position:{{{x}, {y}}}, width:{w}, height:{h}}}'], h
def box(x, y, w, h, txt):
    return [f"set sh to make new shape with properties {{position:{{{x}, {y}}}, width:{w}, height:{h}, object text:{q(txt)}}}",
            "set size of object text of sh to 18"]
def line(x1, y1, x2, y2):
    return [f"make new line with properties {{start point:{{{x1}, {y1}}}, end point:{{{x2}, {y2}}}}}"]

def artifacts_right(stack, y=190):
    out = []
    for kind, payload, cap in stack:
        if kind == "table": cmds, h = table(XR, y, WR, payload)
        elif kind == "tokens": cmds, h = table(XR, y, WR, payload, rh=80, colw=[140, 350, 350])
        elif kind == "image": cmds, h = image(XR, y, payload[0], payload[1])
        else: cmds, h = box(XR, y, WR, payload[1], payload[0]), payload[1]
        out += cmds; y += h + 12
        out += text_item(XR, y, WR, cap); y += 95
    return out

def slide_cmds(sd, art, bullets_box=(X0, 190, 880, 840), extra=()):
    x, y, w, h = bullets_box
    out = ['set s to make new slide with properties {base slide:master slide "Title & Bullets"}', "tell s",
           f"set object text of default title item to {q(sd['title'])}",
           "set size of object text of default title item to 44",
           f"set object text of default body item to {q(chr(10).join(sd['bullets']))}",
           "set size of object text of default body item to 20",
           f"set width of default body item to {w}", f"set height of default body item to {h}",
           f"set position of default body item to {{{x}, {y}}}",
           'set presenter notes to ""']
    return out + list(extra) + art + ["end tell"]

s1, s2, s3, s4, s5, s6 = slides
L += slide_cmds(s1, artifacts_right([("table", s1["tables"][0], s1["caps"][0]), ("tokens", s1["tables"][1], s1["caps"][1])], y=190))
L += slide_cmds(s2, artifacts_right([("table", s2["tables"][0], s2["caps"][0]), ("table", s2["tables"][1], s2["caps"][1])], y=170))
# slide 3: architecture across the top, bullets bottom-left, table bottom-right
arch = []
for yy, lab in [(190, "Premise"), (300, "Hypothesis")]:
    arch += text_item(80, yy + 12, 170, lab, size=20, grey=False)
    arch += box(260, yy, 180, 60, "GloVe 300d") + box(480, yy, 200, 60, "tanh proj. 100d")
    arch += line(440, yy + 30, 480, yy + 30) + line(680, yy + 30, 730, yy + 30)
arch += box(730, 190, 260, 170, "Encoder (shared)\nsum / mean / max\nLSTM / GRU")
arch += line(990, 275, 1040, 275) + box(1040, 245, 200, 60, "concat [u, v]") + line(1240, 275, 1280, 275)
arch += box(1280, 245, 200, 60, "MLP 3 × 200 tanh") + line(1480, 275, 1520, 275) + box(1520, 245, 220, 60, "softmax E / N / C")
arch += text_item(80, 380, 1700, s3["caps"][0])
t3, h3 = table(1000, 480, 840, s3["tables"][0])
L += slide_cmds(s3, t3 + text_item(1000, 480 + h3 + 12, 840, s3["caps"][1]), bullets_box=(X0, 470, 880, 580), extra=arch)
L += slide_cmds(s4, artifacts_right([("image", (f"{FIG}/training_curve.png", 400), s4["caps"][0]),
                                     ("box", ("Placeholder: learning curve\naccuracy vs training-set size", 230), s4["caps"][1])], y=170))
L += slide_cmds(s5, artifacts_right([("image", (f"{FIG}/rope_decisions.png", 400), s5["caps"][0]),
                                     ("table", s5["tables"][0], s5["caps"][1])], y=170))
L += slide_cmds(s6, artifacts_right([("table", s6["tables"][0], s6["caps"][0]),
                                     ("image", (f"{FIG}/bda_weights.png", 470), s6["caps"][-1])], y=170))
L += ["tell slide 1"] + text_item(80, 1000, 1000, "Code: https://github.com/neelshahx/nlp1-2026/tree/main/nli", size=18) + ["end tell"]
for n in range(6):  # slide numbers on content slides (title excluded)
    L += [f"tell slide {n + 2}"] + text_item(1740, 1010, 120, f"{n + 1}/6", size=16) + ["end tell"]
L += ["repeat with i from 1 to count of slides", 'tell slide i to delete (every text item whose object text is "")', "end repeat",
      "end tell", f'save d in POSIX file "{OUT}"', f'export d to POSIX file "{OUT.replace(".key", ".pdf")}" as PDF', "end tell"]
open("/tmp/project2_deck.applescript", "w").write("\n".join(L))
r = subprocess.run(["osascript", "/tmp/project2_deck.applescript"], capture_output=True, text=True)
print(r.stdout, r.stderr); sys.exit(r.returncode)
