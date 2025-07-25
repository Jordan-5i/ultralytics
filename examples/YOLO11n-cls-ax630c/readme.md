AX630C上板运行demo
1. cd `examples/YOLO11n-cls-ax630c`，运行 `python export.py`，会导出onnx模型`examples/YOLO11n-cls-ax630c/yolo11n-cls.onnx`
2. 运行`python run_onnx.py`，验证onnx推理逻辑正确性
3. 用pulsar2编译onnx模型，（hebing修了一个group conv的问题）
    - 编译的配置文件已提供config.json
4. 上板运行 `python run_axmodel.py`
    - 将`ultralytics/cfg/datasets/ImageNet.yaml`文件scp到板子上
