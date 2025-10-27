#!/usr/bin/env python3
# 批量评测脚本 - 用于Fox数据集Tiny/Small两档评测
import os
import sys
import argparse
import json
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image
import time

def setup_model():
    """加载DeepSeek-OCR模型"""
    model_name = 'deepseek-ai/DeepSeek-OCR'
    print(f"加载模型: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, 
        _attn_implementation='flash_attention_2', 
        trust_remote_code=True, 
        use_safetensors=True
    )
    model = model.eval().cuda().to(torch.bfloat16)
    
    return model, tokenizer

def process_image(model, tokenizer, image_path, prompt, base_size, image_size, crop_mode):
    """处理单张图像"""
    try:
        result = model.infer(
            tokenizer, 
            prompt=prompt, 
            image_file=str(image_path), 
            output_path="/tmp",  # 临时输出目录
            base_size=base_size, 
            image_size=image_size, 
            crop_mode=crop_mode, 
            save_results=False, 
            test_compress=False
        )
        return result
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return None

def batch_evaluate(input_dir, output_dir, mode, image_size, prompt, temperature=0, top_p=1):
    """批量评测函数"""
    print(f"开始批量评测: {mode} 模式, 图像尺寸: {image_size}x{image_size}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"提示词: {prompt}")
    
    # 设置参数
    if mode == "tiny":
        base_size = 512
        crop_mode = False
    elif mode == "small":
        base_size = 640
        crop_mode = False
    else:
        raise ValueError(f"不支持的模式: {mode}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    model, tokenizer = setup_model()
    
    # 获取所有图像文件
    input_path = Path(input_dir)
    image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 批量处理
    results = []
    start_time = time.time()
    
    for i, image_file in enumerate(image_files):
        print(f"处理 {i+1}/{len(image_files)}: {image_file.name}")
        
        result = process_image(
            model, tokenizer, image_file, prompt, 
            base_size, image_size, crop_mode
        )
        
        if result:
            results.append({
                "image": image_file.name,
                "result": result,
                "mode": mode,
                "image_size": image_size,
                "base_size": base_size,
                "crop_mode": crop_mode,
                "prompt": prompt
            })
    
    # 保存结果
    output_file = Path(output_dir) / f"fox_{mode}_{image_size}_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    end_time = time.time()
    print(f"评测完成!")
    print(f"处理了 {len(results)} 个文件")
    print(f"耗时: {end_time - start_time:.2f} 秒")
    print(f"结果保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="DeepSeek-OCR Fox数据集批量评测")
    parser.add_argument("--input_dir", required=True, help="输入图像目录")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--mode", choices=["tiny", "small"], required=True, help="评测模式")
    parser.add_argument("--image_size", type=int, required=True, help="图像尺寸")
    parser.add_argument("--prompt", default="Free OCR.", help="提示词")
    parser.add_argument("--temperature", type=float, default=0, help="温度参数")
    parser.add_argument("--top_p", type=float, default=1, help="top_p参数")
    
    args = parser.parse_args()
    
    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    # 运行批量评测
    batch_evaluate(
        args.input_dir, 
        args.output, 
        args.mode, 
        args.image_size, 
        args.prompt, 
        args.temperature, 
        args.top_p
    )

if __name__ == "__main__":
    main()
