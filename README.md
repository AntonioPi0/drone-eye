# Drone-Eye: Edge AI Optimization for Autonomous Landing 🚁

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

**Drone-Eye** is a high-performance computer vision pipeline designed for autonomous drone landing and obstacle detection. This project demonstrates how to bridge the gap between heavy Deep Learning models and resource-constrained **Edge hardware**.

By utilizing **PyTorch FX Graph Mode Quantization**, I reduced model latency and memory footprint significantly while maintaining decision integrity verified through **Explainable AI (XAI)**.

## 📊 Performance Benchmark (Actual Results)
Tests conducted on a MobileNetV2 architecture comparing the FP32 Baseline vs. the INT8 Optimized version.

| Metric | Baseline (FP32) | Optimized (INT8 FX) | Improvement |
| :--- | :--- | :--- | :--- |
| **Model Size** | 13.58 MB | **3.74 MB** | **-72.5%** |
| **Avg Latency** | 40.48 ms | **17.97 ms** | **-55.6%** |
| **Throughput** | ~24.7 FPS | **~55.6 FPS** | **+2.25x Speedup** |
| **XAI Fidelity** | 1.0000 | **0.8775** | (Cosine Similarity) |

## 🚀 Key Technical Features
- **Post-Training Quantization (PTQ):** Implementation of 8-bit quantization using the modern **FX Graph Mode** to handle complex skip-connections.
- **Explainable AI (Grad-CAM):** Integrated activation mapping to ensure the model maintains focus on landing targets post-optimization.
- **Data-Driven Calibration:** Improved fidelity score from 0.65 to 0.88 by calibrating the model with domain-specific drone landing imagery.
- **Edge Deployment Ready:** Optimized for high-frequency control loops (>50Hz), essential for flight safety.

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/drone-eye.git](https://github.com/your-username/drone-eye.git)
cd drone-eye
```

```bash
### 2. Setup the environment

# Create a virtual environment
python -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3. Run the optimization pipeline
The demo_optimization.py script executes the full workflow: loading the FP32 model, performing FX Graph Mode Quantization, running benchmarks, and generating XAI heatmaps.

```bash
python demo_optimization.py
``

```bash
python demo_optimization.py
```
