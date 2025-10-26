#!/usr/bin/env python3
"""
DeepSeek-OCR 推理测试脚本
详细测试流程和输出结果说明
"""

import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_inference(image_path, output_dir, prompt_type="markdown"):
    """
    测试DeepSeek-OCR模型推理
    
    Args:
        image_path: 输入图片路径
        output_dir: 输出目录
        prompt_type: 提示词类型 (markdown/free/detection)
    """
    
    print("🚀 开始加载模型...")
    
    # 步骤1: 加载模型和分词器
    model_name = "deepseek-ai/DeepSeek-OCR"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        _attn_implementation='flash_attention_2'
    )
    
    print("✅ 模型加载完成！")
    
    # 步骤2: 设置提示词
    prompts = {
        "markdown": "<image>\n<|grounding|>Convert the document to markdown.",
        "free": "<image>\n<|grounding|>Extract all text and images from this document.",
        "detection": "<image>\n<|grounding|>Detect all text regions and images in this document."
    }
    
    prompt = prompts.get(prompt_type, prompts["markdown"])
    print(f"📝 使用提示词: {prompt}")
    
    # 步骤3: 执行推理
    print(f"🖼️  处理图片: {image_path}")
    
    result = model.infer(
        prompt=prompt,
        image_file=image_path,
        output_path=output_dir,
        base_size=640,      # 基础尺寸
        image_size=1280,    # 图像尺寸
        crop_mode="dynamic", # 动态裁剪模式
        save_results=True,   # 保存结果
        test_compress=True   # 测试压缩
    )
    
    print("✅ 推理完成！")
    print(f"📂 结果保存在: {output_dir}")
    
    return result

def analyze_output(output_dir):
    """
    分析输出结果
    """
    print("\n" + "="*50)
    print("📊 输出结果分析")
    print("="*50)
    
    # 检查输出文件
    result_mmd = os.path.join(output_dir, "result.mmd")
    result_boxes = os.path.join(output_dir, "result_with_boxes.jpg")
    images_dir = os.path.join(output_dir, "images")
    
    print(f"📄 Markdown结果: {result_mmd}")
    if os.path.exists(result_mmd):
        with open(result_mmd, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"   - 文件大小: {len(content)} 字符")
            print(f"   - 包含图像引用: {content.count('![](')} 个")
    
    print(f"🖼️  标注图像: {result_boxes}")
    if os.path.exists(result_boxes):
        size = os.path.getsize(result_boxes)
        print(f"   - 文件大小: {size/1024:.1f} KB")
    
    print(f"📁 提取图像目录: {images_dir}")
    if os.path.exists(images_dir):
        images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        print(f"   - 提取图像数量: {len(images)} 个")
        for img in sorted(images):
            img_path = os.path.join(images_dir, img)
            size = os.path.getsize(img_path)
            print(f"   - {img}: {size/1024:.1f} KB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek-OCR 推理测试")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--prompt", choices=["markdown", "free", "detection"], 
                       default="markdown", help="提示词类型")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 执行测试
    result = test_inference(args.image, args.output, args.prompt)
    
    # 分析结果
    analyze_output(args.output)
