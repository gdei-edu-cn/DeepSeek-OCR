#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 批量评测脚本 - Tiny(64)/Small(100) 两档；离线优先；逐条落盘
import os, sys, json, time, argparse
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image

TORCH_DTYPE = torch.bfloat16
PROMPT_DEFAULT = "Free OCR."

def _normalize(out):
    if isinstance(out, str):
        return out.strip()
    if isinstance(out, dict):
        # 常见字段兜底
        return (out.get("text") or out.get("generated_text") or out.get("answer") or "").strip()
    if isinstance(out, list) and out:
        o = out[0]
        if isinstance(o, str):  return o.strip()
        if isinstance(o, dict): return (o.get("text") or o.get("generated_text") or "").strip()
    return ""

def setup_model():
    """加载DeepSeek-OCR模型 - 离线优先"""
    model_id = os.getenv("DEEPSEEK_OCR_HF", "deepseek-ai/DeepSeek-OCR")
    local_only = os.path.isdir(model_id) or os.getenv("HF_HUB_OFFLINE", "") == "1"
    print(f"加载模型: {model_id} (离线={local_only})")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=local_only)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_only,
        use_safetensors=True,
        torch_dtype=TORCH_DTYPE,
        # 如果 flash-attn2 报 dtype/环境错误，直接去掉这一行
        _attn_implementation=os.getenv("DSK_FLASH", "flash_attention_2"),
    ).eval().to("cuda")
    return model, tokenizer


def batch_evaluate(input_dir, output_dir, mode, image_size, prompt, temperature=0, top_p=1):
    """批量评测函数 - 离线优先，逐条落盘"""
    print(f"开始批量评测: {mode} 模式, 图像尺寸: {image_size}x{image_size}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"提示词: {prompt}")
    
    # 设置参数（保留以兼容，但实际由 model.chat 内部处理）
    if mode == "tiny":
        base_size = 512
    elif mode == "small":
        base_size = 640
    else:
        raise ValueError(f"不支持的模式: {mode}")
    
    # 创建输出目录和结果文件
    os.makedirs(output_dir, exist_ok=True)
    preds_path = Path(output_dir) / "preds.jsonl"
    if preds_path.exists():
        preds_path.unlink()  # 清空旧结果
    
    # 加载模型
    model, tokenizer = setup_model()
    
    # 获取所有图像文件
    img_dir = Path(input_dir)
    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    print(f"找到 {len(imgs)} 个图像文件")
    
    # 批量处理 - 逐条落盘
    start_time = time.time()
    
    with torch.no_grad(), open(preds_path, "a", encoding="utf-8") as fout:
        for i, img in enumerate(imgs, 1):
            print(f"处理 {i}/{len(imgs)}: {img.name}")
            try:
                out = model.infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=str(img),
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=False,
                    save_results=False,   # 直接用返回值，不落盘
                    test_compress=False
                )
                pred_text = (out or "").strip()
                if not pred_text:
                    raise RuntimeError("infer() 返回空文本")
                # 方便抽查
                print(pred_text[:120].replace("\n", " "))
                rec = {"image": img.name, "pred": pred_text}
            except Exception as e:
                import traceback
                print(f"处理图像 {img.name} 时出错: {e}")
                rec = {"image": img.name, "pred": "", "error": traceback.format_exc()}

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
    
    dur = time.time() - start_time
    print(f"评测完成!")
    print(f"处理了 {len(imgs)} 个文件")
    print(f"耗时: {dur:.2f} 秒")
    print(f"结果保存到: {preds_path}")

def main():
    parser = argparse.ArgumentParser(description="DeepSeek-OCR Fox数据集批量评测")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["tiny", "small"], required=True)
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument("--prompt", default=PROMPT_DEFAULT)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1)
    args = parser.parse_args()

    # 不要在这里改 CUDA_VISIBLE_DEVICES；用你外面 export 的
    batch_evaluate(args.input_dir, args.output, args.mode, args.image_size,
                   args.prompt, args.temperature, args.top_p)

if __name__ == "__main__":
    main()
