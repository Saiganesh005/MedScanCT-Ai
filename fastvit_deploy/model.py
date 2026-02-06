import pathlib

import timm
import torch

class_names = ["COVID", "Normal", "Bacterial Pneumonia","Viral Pneumonia","Tuberculosis"]


def load_model(model_path: pathlib.Path | None = None):
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
    return model, device
