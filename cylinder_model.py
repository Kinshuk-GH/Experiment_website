from pathlib import Path
import math
import numpy as np
import streamlit as st
import torch
from torch import nn
import matplotlib.pyplot as plt

# Must match the architecture and scaling used in Cylinder_Flow_DeepONet_LR_Decay.ipynb.
try:
    from scimba_torch.neural_nets.coordinates_based_nets.mlp import GenericMLP
except ImportError as exc:
    raise ImportError(
        "This app needs the same Scimba Torch package used to train the model. "
        "Install the project's Scimba Torch dependency in the environment, then restart."
    ) from exc



# Physical properties: approximate dry air at 20 °C.
RHO_AIR = 1.204
MU_AIR = 1.825e-5
NU_AIR = MU_AIR / RHO_AIR
D_M = 0.01
XMIN, XMAX, YMIN, YMAX = -20.0, 40.0, -20.0, 20.0
RADIUS = 0.5
RE_MIN, RE_MAX = 1.0, 10.0
BRANCH_WIDTHS = [64] * 4
TRUNK_WIDTHS = [64] * 6
LATENT = 64
N_OUT = 2
DTYPE = torch.float64
MODEL_PATH = Path(__file__).resolve().parent / "cylinder_air_deeponet.pt"

def make_mlp(n_in, n_out, widths):
    return GenericMLP(n_in, n_out, layer_sizes=widths, activation_type="tanh")

class DeepONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = make_mlp(3, N_OUT * LATENT, BRANCH_WIDTHS)
        self.trunk = make_mlp(2, N_OUT * LATENT, TRUNK_WIDTHS)
        self.bias = nn.Parameter(torch.zeros(N_OUT, dtype=DTYPE))

    def branch_features(self, re):
        r = 2.0 * (re - RE_MIN) / (RE_MAX - RE_MIN) - 1.0
        inv = 2.0 * (1.0 / re - 1.0 / RE_MAX) / (1.0 / RE_MIN - 1.0 / RE_MAX) - 1.0
        lg = 2.0 * (torch.log(re) - np.log(RE_MIN)) / (np.log(RE_MAX) - np.log(RE_MIN)) - 1.0
        return torch.cat([r, inv, lg], dim=1)

    def forward(self, xy_norm, re):
        b = self.branch(self.branch_features(re)).view(-1, N_OUT, LATENT)
        t = torch.tanh(self.trunk(xy_norm)).view(-1, N_OUT, LATENT)
        return (b * t).sum(-1) / np.sqrt(LATENT) + self.bias

class CylinderDeepONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = DeepONet()
        self.register_buffer("lower", torch.tensor([XMIN, YMIN, RE_MIN], dtype=DTYPE))
        self.register_buffer("upper", torch.tensor([XMAX, YMAX, RE_MAX], dtype=DTYPE))

    def streamfunction_pressure(self, z):
        x, y = z[:, 0:1], z[:, 1:2]
        q = 2.0 * (z - self.lower) / (self.upper - self.lower) - 1.0
        raw = self.net(q[:, 0:2], z[:, 2:3])
        npsi, np_ = raw[:, 0:1], raw[:, 1:2]
        dc = x.square() + y.square() - RADIUS**2
        xi = (x - XMIN) / (XMAX - XMIN)
        eta = 2.0 * (y - YMIN) / (YMAX - YMIN) - 1.0
        do = xi * (1.0 - eta.square())
        denominator = dc + do
        g = dc / denominator
        b = dc * do / denominator
        H = 3.0 * g.square() - 2.0 * g**3
        psi = y * H + b.square() * npsi
        p = ((XMAX - x) / (XMAX - XMIN)) * np_
        return psi, p

    def forward(self, z):
        outer_grad_enabled = torch.is_grad_enabled()
        with torch.enable_grad():
            if not z.requires_grad:
                z = z.detach().clone().requires_grad_(True)
            psi, p = self.streamfunction_pressure(z)
            dpsi = torch.autograd.grad(
                psi, z, grad_outputs=torch.ones_like(psi), create_graph=True
            )[0]
            u, v = dpsi[:, 1:2], -dpsi[:, 0:1]
            result = torch.cat([u, v, p], dim=1)
        return result if outer_grad_enabled else result.detach()

@st.cache_resource
def load_model(model_path, modified_time):
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Copy cylinder_air_deeponet.pt "
            "into the same folder as app.py. The app intentionally does not train a model."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CylinderDeepONet().to(device=device, dtype=DTYPE)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, device, checkpoint

def predict_field(model, device, re_value, x_range, y_range, nx=241, ny=121):
    x = np.linspace(*x_range, nx)
    y = np.linspace(*y_range, ny)
    XX, YY = np.meshgrid(x, y)
    fluid = XX**2 + YY**2 > RADIUS**2
    pts = np.column_stack([XX[fluid], YY[fluid], np.full(fluid.sum(), re_value)])
    preds = []
    with torch.enable_grad():
        for chunk in np.array_split(pts, max(1, math.ceil(len(pts) / 4096))):
            z = torch.tensor(chunk, device=device, dtype=DTYPE, requires_grad=True)
            preds.append(model(z).detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    u = np.full(XX.shape, np.nan); v = np.full(XX.shape, np.nan); p = np.full(XX.shape, np.nan)
    u[fluid], v[fluid], p[fluid] = pred[:, 0], pred[:, 1], pred[:, 2]
    speed = np.hypot(u, v) * float(U_inf(re_value))
    pressure_pa = p * float(pressure_scale(re_value))
    return XX, YY, u, v, speed, pressure_pa

def U_inf(re):
    return re * NU_AIR / D_M

def pressure_scale(re):
    return MU_AIR * U_inf(re) / D_M

