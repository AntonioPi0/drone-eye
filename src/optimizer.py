import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
import copy

class DroneModelOptimizer:
    def __init__(self):
        # fbgemm per laptop, qnnpack per Jetson/ARM
        self.engine = 'fbgemm'
        self.qconfig = get_default_qconfig(self.engine)

    def optimize(self, model_fp32, calibration_data=None): # Aggiungi questo parametro
        model_fp32.eval()
        model_to_quantize = copy.deepcopy(model_fp32)
        qconfig_dict = {"": self.qconfig}
        example_inputs = torch.randn(1, 3, 224, 224)
        model_prepared = prepare_fx(model_to_quantize, qconfig_dict, example_inputs)

        print("Calibrazione in corso con dati reali...")
        with torch.no_grad():
            if calibration_data is not None:
                for tensor in calibration_data:
                    if tensor.ndim == 3:
                        tensor = tensor.unsqueeze(0)
                    model_prepared(tensor) # Calibra su immagini vere!
            else:
                # Fallback se non ci sono dati
                for _ in range(20):
                    model_prepared(torch.randn(1, 3, 224, 224))

        model_int8 = convert_fx(model_prepared)
        return model_int8
