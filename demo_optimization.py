import torch
import time
import os
import cv2
import numpy as np
from torchvision import transforms
from src.vision_engine import DroneVisionEngine
from src.optimizer import DroneModelOptimizer

def get_size_mb(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / (1024 * 1024)
    os.remove("temp.p")
    return size

def benchmark(model, input_tensor):
    model.eval()
    for _ in range(10):
        with torch.no_grad(): _ = model(input_tensor)

    start = time.time()
    with torch.no_grad():
        for _ in range(50): _ = model(input_tensor)
    end = time.time()
    return (end - start) / 50 * 1000

def run_optimization_test():
    torch.backends.quantized.engine = 'fbgemm'

    engine = DroneVisionEngine()
    optimizer = DroneModelOptimizer()

    # 1. SETUP DATI E MODELLO OTTIMIZZATO
    # Ottimizziamo il modello una sola volta all'inizio
    print(f"--- Generazione Modello Ottimizzato ---")

    size_fp32 = get_size_mb(engine.model)
    # Lista immagini da testare
    img_files = ["test_drone_2.png", "test_drone_3.png", "test_drone_4.png"]
    data_dir = "./data/"

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Crea una lista di tensori dalle tue immagini .webp
    calibration_tensors = []
    for img_name in img_files:
        raw_path = os.path.join("./data/", img_name)
        img = cv2.imread(raw_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Fondamentale: unsqueeze(0) aggiunge la dimensione del batch
            tensor = preprocess(img).unsqueeze(0)
            calibration_tensors.append(tensor)
    model_int8 = optimizer.optimize(engine.model, calibration_data=calibration_tensors)

    size_int8 = get_size_mb(model_int8)

    # Accumulatori per le medie
    all_latencies_fp32 = []
    all_latencies_int8 = []
    all_fidelities = []

    print(f"\n--- Inizio Test su {len(img_files)} immagini ---")

    print(f"\n--- Inizio Test su {len(img_files)} immagini ---")

    for idx, img_name in enumerate(img_files):
        img_path = os.path.join(data_dir, img_name)
        if not os.path.exists(img_path):
            print(f"⚠️ Salto {img_name}: file non trovato.")
            continue

        # Load & Preprocess
        raw_img = cv2.imread(img_path)
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess(raw_img).unsqueeze(0)

        # Benchmarks singoli
        lat_fp32 = benchmark(engine.model, input_tensor)
        lat_int8 = benchmark(model_int8, input_tensor)

        # Fidelity
        with torch.no_grad():
            out_fp32 = engine.model(input_tensor)
            out_int8 = model_int8(input_tensor)
            fid = torch.nn.functional.cosine_similarity(out_fp32, out_int8).mean().item()

        # Salvataggio Heatmap (Baseline)
        heatmap_fp32, _ = engine.get_gradcam(input_tensor)
        res_img = engine.overlay_heatmap(heatmap_fp32, cv2.resize(raw_img, (224,224)))
        cv2.imwrite(f"result_img_{idx+1}.jpg", cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))

        # Accumulo dati
        all_latencies_fp32.append(lat_fp32)
        all_latencies_int8.append(lat_int8)
        all_fidelities.append(fid)
        print(f"✅ Processata immagine {idx+1}/{len(img_files)}: {img_name}")

    # 2. CALCOLO MEDIE E REPORT FINALE
    avg_lat_fp32 = np.mean(all_latencies_fp32)
    avg_lat_int8 = np.mean(all_latencies_int8)
    avg_fid = np.mean(all_fidelities)

    weight_reduction = ((size_fp32 - size_int8) / size_fp32) * 100
    avg_speedup = avg_lat_fp32 / avg_lat_int8
    avg_latency_saved = avg_lat_fp32 - avg_lat_int8

    print(f"\n" + "="*50)
    print(f"🏆 FINAL AGGREGATED REPORT (Average over {len(all_fidelities)} images)")
    print(f"="*50)
    print(f"📦 Model Size: {size_fp32:.2f}MB -> {size_int8:.2f}MB | Reduction: {weight_reduction:.1f}%")
    print(f"⏱️  Avg Latency: {avg_lat_fp32:.2f}ms -> {avg_lat_int8:.2f}ms")
    print(f"🚀 Avg Speedup: {avg_speedup:.2f}")
    print(f"📉 Latency Saved: {avg_latency_saved:.2f}ms per frame")
    print(f"🎯 Avg XAI Fidelity: {avg_fid:.4f}")
    print(f"📸 Heatmaps saved as result_img_1..3.jpg")
    print(f"="*50)

if __name__ == "__main__":
    run_optimization_test()
