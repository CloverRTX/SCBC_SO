import os
import torch
import sympy as sp

def save(model, config, model_type, state):
    folder = os.path.dirname(config["save_load"]["save_path"])
    os.makedirs(folder, exist_ok=True)
    path = config["save_load"]["save_path"].format(model_type = model_type, state=state)
    torch.save(model.state_dict(), path)


def load(model, config, model_type, state):
    path = config["save_load"]["load_path"].format(model_type = model_type, state=state)
    #model.load_state_dict(torch.load(path, map_location=config["device"]))
    model.load_state_dict(torch.load(path))


# 多项式
def sympy_save(model, config, model_type, state):
    folder = os.path.dirname(config["save_load"]["poly_save_path"])
    os.makedirs(folder, exist_ok=True)
    path = config["save_load"]["poly_save_path"].format(model_type=model_type, state=state)
    with open(path, "w") as f:
        f.write(str(model))

def sympy_load(config, model_type, state):
    path = config["save_load"]["poly_load_path"].format(model_type=model_type, state=state)
    with open(path) as f:
        s = f.read()
    return sp.sympify(s)
