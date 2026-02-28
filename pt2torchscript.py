#!/usr/bin/env python3
"""
PT模型转TorchScript格式工具
用于将普通的.pt模型转换为RKNN所需的TorchScript格式
"""
import os
import sys
import argparse

def convert_to_torchscript(input_pt, output_torchscript=None, use_ultralytics=True):
    """
    将PT模型转换为TorchScript格式
    
    Args:
        input_pt: 输入的.pt模型文件
        output_torchscript: 输出的TorchScript文件（可选）
        use_ultralytics: 是否使用ultralytics的方法（适用于YOLO模型）
    """
    if not os.path.exists(input_pt):
        print(f"❌ 错误：模型文件不存在: {input_pt}")
        return False
    
    if output_torchscript is None:
        base_name = os.path.splitext(input_pt)[0]
        output_torchscript = f"{base_name}_torchscript.pt"
    
    print("="*60)
    print("PT to TorchScript 转换工具")
    print("="*60)
    print(f"输入模型: {input_pt}")
    print(f"输出模型: {output_torchscript}")
    print(f"转换方法: {'Ultralytics' if use_ultralytics else 'PyTorch'}")
    print("="*60)
    
    try:
        if use_ultralytics:
            # 方法1: 使用ultralytics（适用于YOLOv8等）
            print("\n正在使用 Ultralytics 导出...")
            try:
                from ultralytics import YOLO
                
                model = YOLO(input_pt)
                print(f"✓ 模型加载成功: {model.model}")
                
                # 导出为TorchScript
                success = model.export(format='torchscript', optimize=True)
                
                # ultralytics会自动生成文件名
                auto_output = os.path.splitext(input_pt)[0] + '.torchscript'
                
                if os.path.exists(auto_output):
                    # 如果指定了不同的输出文件名，重命名
                    if auto_output != output_torchscript:
                        os.rename(auto_output, output_torchscript)
                    print(f"\n✓ 转换成功!")
                    print(f"✓ 输出文件: {output_torchscript}")
                    
                    # 显示文件大小
                    file_size = os.path.getsize(output_torchscript) / 1024 / 1024
                    print(f"✓ 文件大小: {file_size:.2f} MB")
                    return True
                else:
                    print("❌ 输出文件未生成")
                    return False
                    
            except ImportError:
                print("⚠️  Ultralytics 未安装，尝试使用PyTorch方法...")
                use_ultralytics = False
        
        if not use_ultralytics:
            # 方法2: 使用PyTorch（通用方法）
            print("\n正在使用 PyTorch 导出...")
            import torch
            
            # 加载模型
            print("加载模型...")
            checkpoint = torch.load(input_pt, map_location='cpu')
            
            # 尝试提取模型
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    print("❌ 错误：检测到state_dict格式")
                    print("   这种格式需要模型架构定义才能转换")
                    print("   建议使用ultralytics的YOLO模型")
                    return False
                else:
                    print("❌ 错误：无法识别的checkpoint格式")
                    return False
            else:
                model = checkpoint
            
            model.eval()
            
            # 创建示例输入
            print("创建示例输入...")
            example_input = torch.randn(1, 3, 640, 640)
            
            # 转换为TorchScript
            print("转换为TorchScript...")
            traced_model = torch.jit.trace(model, example_input)
            
            # 保存
            print(f"保存到: {output_torchscript}")
            traced_model.save(output_torchscript)
            
            print(f"\n✓ 转换成功!")
            print(f"✓ 输出文件: {output_torchscript}")
            
            file_size = os.path.getsize(output_torchscript) / 1024 / 1024
            print(f"✓ 文件大小: {file_size:.2f} MB")
            return True
            
    except Exception as e:
        print(f"\n❌ 转换失败: {str(e)}")
        print("\n可能的原因：")
        print("1. 模型格式不兼容")
        print("2. 缺少模型架构定义")
        print("3. PyTorch版本不匹配")
        print("\n建议：")
        print("- 如果是YOLOv8模型，确保安装了ultralytics")
        print("- 如果是自定义模型，可能需要提供模型定义代码")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='将PT模型转换为TorchScript格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法（适用于YOLO模型）
  python pt2torchscript.py model.pt
  
  # 指定输出文件名
  python pt2torchscript.py model.pt -o model_traced.pt
  
  # 使用PyTorch方法
  python pt2torchscript.py model.pt --pytorch
        """
    )
    
    parser.add_argument('input', help='输入的.pt模型文件')
    parser.add_argument('-o', '--output', help='输出的TorchScript文件（可选）')
    parser.add_argument('--pytorch', action='store_true', 
                       help='使用PyTorch方法而不是Ultralytics')
    
    args = parser.parse_args()
    
    success = convert_to_torchscript(
        args.input, 
        args.output, 
        use_ultralytics=not args.pytorch
    )
    
    if success:
        print("\n" + "="*60)
        print("✓ 转换完成！现在可以使用转换后的模型进行RKNN转换了")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("❌ 转换失败")
        print("="*60)
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("用法: python pt2torchscript.py <pt_model_file> [-o output_file] [--pytorch]")
        print("示例: python pt2torchscript.py model.pt")
        print("帮助: python pt2torchscript.py -h")
        sys.exit(1)
    
    main()
