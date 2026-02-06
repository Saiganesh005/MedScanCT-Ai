import pathlib

import timm
import torch

class_names = ["COVID", "Normal", "Viral Pneumonia","Bacterial Pneumonia","Tuberculosis"]

def load_model(model_path: pathlib.Path | None = None, *, include_target_layer: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_path = model_path or pathlib.Path(__file__).with_name("fastvit_covid_ct.pth")

    model = timm.create_model(
        "fastvit_t8",
        pretrained=False,
        num_classes=len(class_names),
    )

    model.load_state_dict(torch.load(resolved_path, map_location=device))
    model.to(device)
    model.eval()

    if include_target_layer:
        target_layer = model.blocks[-1]
        return model, device, target_layer

    return model, device
