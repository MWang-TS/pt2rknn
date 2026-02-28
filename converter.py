"""
PT to RKNN Model Converter
基于成功转换脚本的核心转换逻辑
"""
import os
import sys
import torch
from rknn.api import RKNN


class PT2RKNNConverter:
    """PT模型到RKNN模型的转换器（支持自动转换TorchScript）"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.rknn = None
    
    def pt_to_torchscript(self, pt_model_path, output_path=None, input_size=(640, 640)):
        """
        将普通PT模型转换为TorchScript格式
        
        Args:
            pt_model_path: PT模型文件路径
            output_path: 输出TorchScript文件路径
            input_size: 输入尺寸 (height, width)
            
        Returns:
            (success, message, torchscript_path)
        """
        try:
            self._log(f"检测到普通PT模型，正在转换为TorchScript...")
            
            # 设置输出路径
            if not output_path:
                base_name = os.path.splitext(pt_model_path)[0]
                output_path = f"{base_name}_rknnopt.torchscript"
            
            # 加载PT模型
            self._log(f"加载模型: {pt_model_path}")
            is_ultralytics = False
            try:
                # 尝试使用ultralytics
                from ultralytics import YOLO
                model = YOLO(pt_model_path)
                model_obj = model.model
                is_ultralytics = True
                self._log("✓ 使用ultralytics加载模型成功")
            except Exception as e1:
                self._log(f"ultralytics加载失败，尝试直接加载: {e1}")
                try:
                    # 尝试直接用torch加载
                    checkpoint = torch.load(pt_model_path, map_location='cpu')
                    if isinstance(checkpoint, dict) and 'model' in checkpoint:
                        model_obj = checkpoint['model']
                        if hasattr(model_obj, 'float'):
                            model_obj = model_obj.float()
                    else:
                        model_obj = checkpoint
                    self._log("✓ 使用torch直接加载成功")
                except Exception as e2:
                    return False, f"无法加载模型。ultralytics错误: {e1}, torch错误: {e2}", None
            
            # 设置为评估模式
            model_obj.eval()
            
            # 对于ultralytics模型，设置为导出模式
            if is_ultralytics:
                self._log("设置ultralytics模型为导出模式...")
                # 禁用动态操作，使模型更容易trace
                for m in model_obj.modules():
                    # 设置Detect层的导出模式
                    if hasattr(m, 'export'):
                        m.export = True
                        # 必须同时设置format属性
                        if not hasattr(m, 'format'):
                            m.format = 'torchscript'
                        self._log(f"  - 设置 {m.__class__.__name__}.export = True, format = torchscript")
                    # 禁用动态anchor生成
                    if hasattr(m, 'dynamic'):
                        m.dynamic = False
                        self._log(f"  - 设置 {m.__class__.__name__}.dynamic = False")
                    # 设置为推理模式
                    if hasattr(m, 'inplace'):
                        m.inplace = False
                        self._log(f"  - 设置 {m.__class__.__name__}.inplace = False")
            
            # 创建示例输入
            self._log(f"创建示例输入: [1, 3, {input_size[0]}, {input_size[1]}]")
            dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
            
            # 先执行一次前向传播，确保所有动态层初始化
            self._log("预热模型（初始化动态层）...")
            with torch.no_grad():
                _ = model_obj(dummy_input)
            
            # 转换为TorchScript (禁用sanity check以避免动态操作问题)
            self._log("使用torch.jit.trace转换（禁用sanity check）...")
            with torch.no_grad():
                traced_model = torch.jit.trace(model_obj, dummy_input, strict=False, check_trace=False)
            
            # 保存
            self._log(f"保存TorchScript模型: {output_path}")
            torch.jit.save(traced_model, output_path)
            
            # 验证
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / 1024 / 1024
                self._log(f"✓ TorchScript转换成功! 文件大小: {file_size:.2f} MB")
                return True, f"TorchScript转换成功", output_path
            else:
                return False, "TorchScript文件未生成", None
                
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            self._log(f"转换错误详情:\n{error_detail}")
            return False, f"PT转TorchScript失败: {str(e)}", None
    
    def check_model_format(self, model_path):
        """
        检查模型格式是否为TorchScript
        
        Returns:
            (is_torchscript, message)
        """
        try:
            # 尝试作为TorchScript加载
            model = torch.jit.load(model_path)
            return True, "模型格式正确（TorchScript）"
        except Exception as e:
            error_msg = str(e)
            if "constants.pkl" in error_msg or "PytorchStreamReader" in error_msg:
                return False, "模型是普通PT格式，需要先转换为TorchScript"
            else:
                return False, f"模型格式检查失败: {error_msg}"
        
    def convert(self, 
                pt_model_path,
                platform='rk3576',
                do_quant=True,
                dataset_path=None,
                output_path=None,
                input_size=(640, 640),
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                optimization_level=3,
                auto_convert_torchscript=True):
        """
        执行PT到RKNN的转换（支持自动转换TorchScript）
        
        Args:
            pt_model_path: PT模型文件路径
            platform: 目标平台 (rk3562/rk3566/rk3568/rk3576/rk3588)
            do_quant: 是否量化 (True=int8, False=fp)
            dataset_path: 校准数据集路径（量化时必需）
            output_path: 输出RKNN文件路径
            input_size: 输入尺寸 (height, width)
            mean_values: 均值
            std_values: 标准差
            optimization_level: 优化等级 (0-3)
            auto_convert_torchscript: 是否自动转换为TorchScript (默认True)
            
        Returns:
            (success, message, output_file)
        """
        torchscript_path = None  # 用于清理临时文件
        
        try:
            # 参数验证
            if not os.path.exists(pt_model_path):
                return False, f"模型文件不存在: {pt_model_path}", None
                
            if platform not in ['rk3562', 'rk3566', 'rk3568', 'rk3576', 'rk3588']:
                return False, f"不支持的平台: {platform}", None
            
            # 检查模型格式
            is_torchscript, format_msg = self.check_model_format(pt_model_path)
            
            # 如果不是TorchScript且允许自动转换
            if not is_torchscript and auto_convert_torchscript:
                self._log(f"⚠️  {format_msg}")
                self._log("🔄 自动转换模式已启用，开始转换...")
                
                # 转换为TorchScript
                success, msg, torchscript_path = self.pt_to_torchscript(
                    pt_model_path, 
                    input_size=input_size
                )
                
                if not success:
                    return False, f"TorchScript转换失败: {msg}", None
                
                # 使用转换后的TorchScript文件
                pt_model_path = torchscript_path
                self._log(f"✓ 将使用转换后的模型: {torchscript_path}")
                
            elif not is_torchscript:
                # 不允许自动转换，返回错误
                error_msg = f"{format_msg}\n\n"
                error_msg += "❌ 错误：RKNN需要TorchScript格式的模型\n\n"
                error_msg += "📝 解决方法：\n"
                error_msg += "1. 启用自动转换（推荐）\n"
                error_msg += "2. 手动导出TorchScript：\n"
                error_msg += "   from ultralytics import YOLO\n"
                error_msg += "   model = YOLO('your_model.pt')\n"
                error_msg += "   model.export(format='torchscript')"
                return False, error_msg, None
            else:
                self._log(f"✓ {format_msg}")
            
            # 量化需要数据集
            if do_quant and not dataset_path:
                return False, "量化模式需要提供校准数据集", None
                
            if do_quant and dataset_path and not os.path.exists(dataset_path):
                return False, f"校准数据集文件不存在: {dataset_path}", None
            
            # 设置输出路径
            if not output_path:
                model_name = os.path.splitext(os.path.basename(pt_model_path))[0]
                # 移除可能的_rknnopt后缀
                model_name = model_name.replace('_rknnopt', '')
                quant_suffix = 'i8' if do_quant else 'fp'
                output_path = f"./output/{model_name}_{platform}_{quant_suffix}.rknn"
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 初始化RKNN
            self._log("初始化RKNN...")
            self.rknn = RKNN(verbose=self.verbose)
            
            # 配置模型
            self._log("配置模型参数...")
            self.rknn.config(
                mean_values=mean_values,
                std_values=std_values,
                target_platform=platform,
                quantized_algorithm='normal',
                quantized_method='channel',
                optimization_level=optimization_level
            )
            
            # 加载模型
            self._log(f"加载TorchScript模型到RKNN: {pt_model_path}")
            ret = self.rknn.load_pytorch(
                model=pt_model_path,
                input_size_list=[[1, 3, input_size[0], input_size[1]]]
            )
            if ret != 0:
                return False, "RKNN加载模型失败", None
            
            # 构建模型
            self._log(f"构建RKNN模型 (量化: {'是' if do_quant else '否'})...")
            ret = self.rknn.build(
                do_quantization=do_quant,
                dataset=dataset_path if do_quant else None,
                rknn_batch_size=1
            )
            if ret != 0:
                return False, "RKNN构建模型失败", None
            
            # 导出模型
            self._log(f"导出RKNN模型: {output_path}")
            ret = self.rknn.export_rknn(output_path)
            if ret != 0:
                return False, "RKNN导出模型失败", None
            
            # 验证输出文件
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / 1024 / 1024
                self._log(f"✓ 转换成功! 文件大小: {file_size:.2f} MB")
                return True, f"转换成功，文件大小: {file_size:.2f} MB", output_path
            else:
                return False, "输出文件未生成", None
                
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            self._log(f"转换错误:\n{error_detail}")
            return False, f"转换过程出错: {str(e)}", None
            
        finally:
            # 清理资源
            if self.rknn:
                self.rknn.release()
                self.rknn = None
            
            # 清理临时TorchScript文件（可选）
            # if torchscript_path and os.path.exists(torchscript_path):
            #     try:
            #         os.remove(torchscript_path)
            #         self._log(f"✓ 清理临时文件: {torchscript_path}")
            #     except:
            #         pass
    
    def _log(self, message):
        """日志输出"""
        if self.verbose:
            print(f"[Converter] {message}")


def main():
    """命令行入口"""
    if len(sys.argv) < 2:
        print("用法: python converter.py <pt_model_path> [platform] [quant_type] [output_path]")
        print("  platform: rk3562/rk3566/rk3568/rk3576/rk3588 (默认: rk3576)")
        print("  quant_type: i8/fp (默认: i8)")
        print("  output_path: 输出文件路径 (可选)")
        print("\n示例: python converter.py model.pt rk3576 i8")
        sys.exit(1)
    
    pt_model = sys.argv[1]
    platform = sys.argv[2] if len(sys.argv) > 2 else 'rk3576'
    quant_type = sys.argv[3] if len(sys.argv) > 3 else 'i8'
    output_path = sys.argv[4] if len(sys.argv) > 4 else None
    
    do_quant = (quant_type == 'i8')
    dataset_path = './calibration_data/calibration.txt' if do_quant else None
    
    converter = PT2RKNNConverter(verbose=True)
    success, message, output_file = converter.convert(
        pt_model_path=pt_model,
        platform=platform,
        do_quant=do_quant,
        dataset_path=dataset_path,
        output_path=output_path
    )
    
    if success:
        print(f"\n✓ 成功: {message}")
        print(f"✓ 输出文件: {output_file}")
        sys.exit(0)
    else:
        print(f"\n✗ 失败: {message}")
        sys.exit(1)


if __name__ == '__main__':
    main()
