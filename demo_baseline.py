import torch
import cv2
import numpy as np
from src.vision_engine import DroneVisionEngine
from torchvision import transforms

def run_demo():
    # Inizializziamo il motore
    engine = DroneVisionEngine()

    # Prepariamo un input dummy (simuliamo una telecamera 224x224)
    # In futuro qui caricherai frame reali dal drone
    dummy_img_np = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(dummy_img_np).unsqueeze(0)

    # Generiamo la heatmap
    heatmap, class_id = engine.get_gradcam(input_tensor)

    # Creiamo l'overlay visivo
    visual_result = engine.overlay_heatmap(heatmap, dummy_img_np)

    # Salviamo il risultato
    cv2.imwrite("baseline_test_xai.jpg", visual_result)
    print(f"Demo completata! Heatmap salvata. Classe predetta: {class_id}")

if __name__ == "__main__":
    run_demo()
