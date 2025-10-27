# 保存为 scripts/fox_select_100.py
import json, os, shutil, argparse, re, unicodedata, random
from pathlib import Path
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--raw_dir", default="data/Fox/raw")
p.add_argument("--out_img", default="data/Fox/en_pages")
p.add_argument("--out_ann", default="data/Fox/annotations/en_page_ocr_100.json")
p.add_argument("--model",   default="deepseek-ai/DeepSeek-OCR")  # 如果你本地有HF权重就用它；否则指向你HF版权重ID
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

raw = Path(args.raw_dir)
img_dir = raw/"focus_benchmark_test"/"en_pdf_png"
ann_fp  = raw/"focus_benchmark_test"/"en_page_ocr.json"
assert img_dir.exists(), f"missing {img_dir}"
assert ann_fp.exists(),  f"missing {ann_fp}"

# 论文口径：用 DeepSeek-OCR 的 tokenizer 重新分词（词表~129k）
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

def norm_text(t):
    # 轻量归一，避免 tokenizer 版本差异导致边界抖动
    t = unicodedata.normalize("NFKC", t or "")
    t = re.sub(r"\s+", " ", t).strip()
    return t

anns = json.load(open(ann_fp, encoding="utf-8"))
candidates = []
for ex in anns:
    # Fox 的 JSON 里一般是 conversations[1]["value"] 作为 GT 文本（详见官方 README 的配对示例）
    conv = ex.get("conversations") or []
    gt = ""
    if len(conv) >= 2:
        gt = conv[1].get("value","")
    gt = norm_text(gt)
    ntok = len(tok(gt).input_ids)
    if 600 <= ntok <= 1300:
        candidates.append((ex["image"], ntok, ex))

# 论文说"恰好 100 页"，若多于/少于 100，一般是 tokenizer 版本或文本规范化差异。
random.seed(args.seed)
candidates.sort(key=lambda x: x[1])  # 稳定
if len(candidates) >= 100:
    selected = candidates[:100]
else:
    raise SystemExit(f"只有 {len(candidates)} 页满足 600–1300 token，请检查 tokenizer/文本规范化。")

# 拷图 + 导出子集标注
os.makedirs(args.out_img, exist_ok=True)
os.makedirs(Path(args.out_ann).parent, exist_ok=True)
sub = []
for name, ntok, ex in selected:
    src = img_dir/name
    dst = Path(args.out_img)/name
    os.makedirs(dst.parent, exist_ok=True)
    shutil.copy2(src, dst)
    sub.append(ex)

json.dump(sub, open(args.out_ann, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"完成：拷贝 100 张到 {args.out_img}，标注写入 {args.out_ann}")

