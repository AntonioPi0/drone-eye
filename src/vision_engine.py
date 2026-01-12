import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np

class DroneVisionEngine:
    def __init__(self, model_name='mobilenet_v2'):
        # 1. Carichiamo il modello (Baseline FP32)
        self.model = models.mobilenet_v2(weights='DEFAULT').eval()
        # Per MobileNetV2, l'ultimo strato convoluzionale è l'indice -1 delle features
        self.target_layer = self.model.features[-1]

        self.gradients = None
        self.activations = None

        # Registrazione degli hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def get_gradcam(self, input_tensor, class_idx=None):
        """Genera la heatmap Grad-CAM per una data immagine."""
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass per ottenere i gradienti rispetto alla classe predetta
        output[0, class_idx].backward()

        # Calcolo dei pesi (Global Average Pooling dei gradienti)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Combinazione lineare delle attivazioni pesate
        grad_cam = torch.sum(weights * self.activations, dim=1).squeeze()

        # ReLU sulla mappa risultante
        grad_cam = torch.max(grad_cam, torch.tensor(0.0))

        # Normalizzazione tra 0 e 1
        grad_cam -= grad_cam.min()
        grad_cam /= (grad_cam.max() + 1e-7) # Evitiamo divisione per zero

        return grad_cam.detach().cpu().numpy(), class_idx

    def overlay_heatmap(self, heatmap, original_img):
        """Applica la heatmap sull'immagine originale per visualizzarla."""
        heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        result = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
        return result
