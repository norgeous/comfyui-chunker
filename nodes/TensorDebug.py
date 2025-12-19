from comfy.comfy_types.node_typing import IO
import torch

class TensorDebug:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            # "required": {
            #     "tensor": ("*",),
            # },
            "required": {"tensor": (IO.ANY, {})},
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "Chunker"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, tensor):
        # YOLO, anything goes!
        return True
    
    def execute(self, tensor):

        ui_values = {
            "type": type(tensor).__name__,
        }

        match tensor:
            case torch.Tensor():
                ui_values["shape"] = tensor.shape
                ui_values["dtype"] = str(tensor.dtype)
                ui_values["min"]   = torch.min(tensor).item()
                ui_values["max"]   = torch.max(tensor).item()
                # ui_values["mean"]  = torch.mean(tensor).item()

            case dict():
                ui_values["keys"] = list(tensor.keys())

                if list(tensor.keys()) == ['waveform', 'sample_rate']:
                    waveform = tensor['waveform']
                    sample_rate = tensor['sample_rate']
                    ui_values["waveform"] = {
                        "type": str(type(waveform)),
                        "shape": waveform.shape,
                        "dtype": str(waveform.dtype),
                        "min": torch.min(waveform).item(),
                        "max": torch.max(waveform).item(),
                        # "mean": torch.mean(waveform).item(),
                    }
                    ui_values["sample_rate"] = sample_rate

            case _:
                pass

        return {
            "ui": {"values": [ui_values]},
        }
