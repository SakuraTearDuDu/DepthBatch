# 快速开始（简体中文）

[English Quick Start](quickstart.md) | **简体中文**

这份文档只覆盖当前 alpha 版本最小、最稳的上手路径：

1. 先用 `fake` backend 验证 CLI、输出目录和 manifest。
2. 再下载官方 DA-V2 Small 权重。
3. 然后跑真实 `pytorch` 推理。
4. 最后导出 ONNX，并用 `onnxruntime` 做部署验证。

如果你想先看项目定位、许可边界和已知限制，请回到 [README.zh-CN.md](../README.zh-CN.md) 与 [license_notes.md](license_notes.md)。

## 1. Smoke Path

先确认包装、CLI 和输出结构是通的。

```powershell
depthbatch infer-images `
  --backend fake `
  --input tests/fixtures/images `
  --output runs/fake-smoke `
  --save-raw `
  --stdout-json
```

检查这次运行：

```powershell
depthbatch inspect --run runs/fake-smoke --stdout-json
```

## 2. 真实 PyTorch 路径

安装真实后端依赖，并把官方 DA-V2 Small 权重下载到仓库内。

```powershell
pip install -e .[pytorch]
python scripts/download_da_v2_small.py
```

然后运行真实 PyTorch 推理：

```powershell
depthbatch infer-images `
  --backend pytorch `
  --model da-v2-small `
  --weights artifacts\weights\depth_anything_v2_vits.pth `
  --input tests\fixtures\images `
  --output runs\da-v2-small `
  --save-raw `
  --stdout-json
```

当前仓库的 canonical 语义基线仍然是原生 `pytorch` 路线。即使后续走 ONNX/TensorRT，验证和排错也建议先从这个路径开始。

当前记录的 Windows/NVIDIA GPU 安装命令如下：

```powershell
pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

这些 GPU wheel 仍然是环境相关选择，因此没有被固定进 `pyproject.toml`。

## 3. ONNX 导出与部署验证

安装 ONNX 相关依赖：

```powershell
pip install -e .[pytorch,onnx]
```

导出 ONNX：

```powershell
depthbatch export-onnx `
  --backend pytorch `
  --model da-v2-small `
  --weights artifacts\weights\depth_anything_v2_vits.pth `
  --output artifacts\onnx `
  --overwrite `
  --stdout-json
```

用 ONNXRuntime 做最小部署验证：

```powershell
depthbatch infer-onnx `
  --model da-v2-small `
  --onnx-path artifacts\onnx\artifacts\export\model.onnx `
  --input tests\fixtures\images `
  --output runs\onnx-infer `
  --overwrite `
  --save-raw `
  --stdout-json
```

当前 alpha 版本对 ONNXRuntime 的已验证路径，是静态方形导出（`input_size=518`）。这表示部署验证链路已经跑通，不表示与 canonical PyTorch 预处理契约逐像素等价。

当前记录的本地 CUDA 路径也已通过 `--device cuda` 实测。

## 4. inspect 与 benchmark

查看一次运行的 manifest 汇总：

```powershell
depthbatch inspect --run runs\da-v2-small --stdout-json
```

对 `pytorch` 与 `onnxruntime` 跑一个最小 benchmark：

```powershell
depthbatch benchmark `
  --model da-v2-small `
  --weights artifacts\weights\depth_anything_v2_vits.pth `
  --onnx-path artifacts\onnx\artifacts\export\model.onnx `
  --input tests\fixtures\images `
  --output runs\benchmark-small `
  --stdout-json
```

benchmark 会输出：

- 后端耗时统计
- `reports/benchmark.json`
- `reports/benchmark.md`

当前实现还会在满足条件时补充 `pytorch` 与 `onnxruntime` 的统计一致性信息，但不把它表述为“数值完全等价保证”。

## 5. 当前边界

- 仓库不把模型权重放进 Git 历史。
- DA-V2 Small 是当前默认、且公开文档重点覆盖的模型路径。
- `fake` 是 smoke-test backend，不是模型质量证明。
- `transformers` 是实验性兼容路径。
- TensorRT 仍然是未来扩展点，而不是本版本交付项。

更多部署边界请看 [deployment_notes.md](deployment_notes.md)。
