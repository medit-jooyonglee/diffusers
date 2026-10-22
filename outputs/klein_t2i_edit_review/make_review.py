"""생성 데이터 육안 확인용 시트: 원본 | 프롬프트(편집, text2img) | 결과. 사용: python make_review.py [데이터 폴더] [개수] [seed]"""
import json, random, sys, textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
D = Path(sys.argv[1] if len(sys.argv) > 1 else "/data1/jooyonglee/public/flux2_klein4b_t2i_edit_17k")
N, SEED = (int(sys.argv[2]) if len(sys.argv) > 2 else 24), (int(sys.argv[3]) if len(sys.argv) > 3 else 0)
OUT = Path(__file__).resolve().parent
done = sorted(p.stem for p in (D / "editimg").glob("*.json"))
picks = sorted(random.Random(SEED).sample(done, min(N, len(done))))
H, TW, PER = 360, 430, 8
big, small, idf = ImageFont.load_default(size=19), ImageFont.load_default(size=13), ImageFont.load_default(size=15)
def fit(im):
    im = im.convert("RGB"); im.thumbnail((int(H * 1.5), H)); return im
rows = []
for sid in picks:
    meta = json.loads((D / "editimg" / f"{sid}.json").read_text())
    src, res = fit(Image.open(D / meta["input_image"])), fit(Image.open(D / "editimg" / f"{sid}.png"))
    t2i_path = D / "text2img" / f"{sid}.txt"
    t2i = t2i_path.read_text().strip() if t2i_path.exists() else "(no text2img prompt)"
    row = Image.new("RGB", (src.width + TW + res.width, H), "white"); d = ImageDraw.Draw(row)
    row.paste(src, (0, (H - src.height) // 2)); row.paste(res, (src.width + TW, (H - res.height) // 2))
    x, y = src.width + 12, 10
    d.text((x, y), f"{sid}  {meta['width']}x{meta['height']}", fill=(90, 90, 90), font=idf); y += 26
    d.text((x, y), "EDIT", fill=(200, 60, 30), font=idf); y += 22
    for line in textwrap.wrap(meta["edit_prompt"], 38)[:6]:
        d.text((x, y), line, fill="black", font=big); y += 24
    y += 12; d.text((x, y), "text2img", fill=(40, 90, 190), font=idf); y += 20
    for line in textwrap.wrap(t2i, 62)[:9]:
        d.text((x, y), line, fill=(70, 70, 70), font=small); y += 16
    ax, ay = src.width + TW // 2, H - 30  # 원본 -> 결과 화살표 (폰트에 화살표 글자가 없어 직접 그린다)
    d.line((ax - 40, ay, ax + 30, ay), fill=(200, 60, 30), width=5)
    d.polygon([(ax + 30, ay - 12), (ax + 50, ay), (ax + 30, ay + 12)], fill=(200, 60, 30))
    rows.append(row)
sheets = (len(rows) + PER - 1) // PER
for k in range(sheets):
    chunk = rows[k * PER:(k + 1) * PER]; W = max(r.width for r in chunk)
    sheet = Image.new("RGB", (W, (H + 6) * len(chunk) + 40), (235, 235, 235)); d = ImageDraw.Draw(sheet)
    d.text((10, 10), f"original  |  prompts  |  edited      ({D.name}, {len(done)} done, sheet {k + 1}/{sheets})", fill="black", font=idf)
    y = 40
    for r in chunk:
        sheet.paste(r, (0, y)); y += H + 6
    sheet.save(OUT / f"review_sheet_{k + 1}.jpg", quality=90)
(OUT / "review_ids.txt").write_text("\n".join(picks) + "\n")
print(f"{len(done)} done, {len(picks)} picked -> {sheets} sheets in {OUT}")
