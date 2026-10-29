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

st.set_page_config(page_title="Thermofluid Lab | Cylinder Flow", page_icon="💧", layout="wide")

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
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
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

st.title("💧 Thermofluid Lab")
st.caption("Virtual experiments in fluid mechanics and heat transfer")
st.markdown("""
This lab uses a **pre-trained physics-informed neural network** to estimate how air flows around a circular cylinder.
Choose a Reynolds number and explore the predicted flow field. The model is loaded from a `.pt` checkpoint;
**no neural-network training takes place when you use the website**.
""")

with st.sidebar:
    st.header("Experiment controls")
    re_value = st.slider("Reynolds number (Re)", min_value=1.0, max_value=10.0, value=5.0, step=0.5,
                         help="Re compares the effect of inertia with the effect of viscosity.")
    view = st.selectbox("Region to view", ["Near the cylinder", "Larger region"])
    st.divider()
    st.caption("The current model was trained for Re = 1–10. Results outside this range should not be used.")

st.header("Experiment 1: Air flowing past a cylinder")
st.subheader("What are we investigating?")
st.write("""
Imagine air moving past a round pole. The air has to move around the pole, and its speed and pressure change.
This experiment uses a computer model to estimate those changes. A neural network is a mathematical pattern-finder;
here it has also been taught to respect equations that describe fluid flow.
""")
with st.expander("The science, explained simply"):
    st.markdown("""
    - **Fluid:** a substance that can flow, such as air or water.
    - **Cylinder:** the round object placed in the flow.
    - **Velocity:** how fast the air moves and which direction it travels.
    - **Pressure:** how strongly the air pushes on a surface.
    - **Reynolds number (Re):** a number that compares the tendency of moving air to keep going with the tendency of viscosity to smooth the motion. A low Re usually means viscosity has a stronger influence.
    - **Physics-informed neural network:** a computer model trained to make predictions while also following the mathematical rules of fluid flow.

    The cylinder is stationary. The air enters from the left. The model predicts two velocity components and gauge pressure.
    Colours in the plots show predicted values; they are not photographs of real air.
    """)
st.markdown("**Model assumptions:** steady, two-dimensional flow; approximate dry-air properties at 20 °C; cylinder diameter assumed to be 1 cm; model domain extends 20 diameters upstream and 40 diameters downstream.")

if not MODEL_PATH.exists():
    st.error(f"Checkpoint missing: `{MODEL_PATH.name}`")
    st.info("Place the trained `cylinder_air_deeponet.pt` file beside `app.py`. You do not need to run the training notebook to use this app.")
    st.stop()

try:
    model, device, checkpoint = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as exc:
    st.error("The saved model could not be loaded. Confirm that the checkpoint was produced by the supplied notebook and that the Scimba Torch version is compatible.")
    st.exception(exc)
    st.stop()

x_range, y_range = ((-4, 12), (-4, 4)) if view == "Near the cylinder" else ((-20, 40), (-20, 20))
with st.spinner("Calculating the flow field using the saved model..."):
    XX, YY, u, v, speed, pressure_pa = predict_field(model, device, re_value, x_range, y_range)

u_inf = float(U_inf(re_value))
c1, c2, c3 = st.columns(3)
c1.metric("Reynolds number", f"{re_value:g}")
c2.metric("Inlet air speed (assumed)", f"{u_inf:.4f} m/s")
c3.metric("Approx. cylinder diameter", f"{D_M*100:.1f} cm")

tabs = st.tabs(["Speed", "Pressure", "Flow direction"])
with tabs[0]:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.contourf(XX, YY, np.ma.masked_invalid(speed), levels=32)
    fig.colorbar(im, ax=ax, label="Speed (m/s)")
    ax.add_patch(plt.Circle((0, 0), RADIUS, color="lightgray", ec="black", zorder=4))
    ax.set(xlabel="Distance along flow (cylinder diameters)", ylabel="Distance across flow (cylinder diameters)",
           title=f"Predicted air speed at Re = {re_value:g}")
    ax.set_aspect("equal"); ax.grid(alpha=.15)
    st.pyplot(fig); plt.close(fig)
    st.caption("A colour scale shows the predicted speed. The gray circle is the solid cylinder.")
with tabs[1]:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.contourf(XX, YY, np.ma.masked_invalid(pressure_pa), levels=32)
    fig.colorbar(im, ax=ax, label="Gauge pressure (Pa)")
    ax.add_patch(plt.Circle((0, 0), RADIUS, color="lightgray", ec="black", zorder=4))
    ax.set(xlabel="Distance along flow (cylinder diameters)", ylabel="Distance across flow (cylinder diameters)",
           title=f"Predicted gauge pressure at Re = {re_value:g}")
    ax.set_aspect("equal"); ax.grid(alpha=.15)
    st.pyplot(fig); plt.close(fig)
    st.caption("Gauge pressure is pressure relative to the model's zero-pressure reference at the outlet.")
with tabs[2]:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.streamplot(XX[0, :], YY[:, 0], np.nan_to_num(u, nan=0.0), np.nan_to_num(v, nan=0.0), density=1.1, linewidth=.7, arrowsize=1)
    ax.add_patch(plt.Circle((0, 0), RADIUS, color="lightgray", ec="black", zorder=4))
    ax.set(xlabel="Distance along flow (cylinder diameters)", ylabel="Distance across flow (cylinder diameters)",
           title=f"Predicted flow paths at Re = {re_value:g}")
    ax.set_aspect("equal"); ax.grid(alpha=.15)
    st.pyplot(fig); plt.close(fig)
    st.caption("Streamlines indicate the local direction of motion; they are computed from the predicted velocity components.")

st.subheader("Try an investigation")
st.markdown(f"""
1. Keep the region set to **Near the cylinder**.
2. Change Re from **1** to **10**.
3. Compare how the speed colours, pressure colours, and flow paths change.
4. Write down one observation and one question you would investigate further.
""")
st.info("These are model predictions, not a substitute for measurements. Accuracy depends on the trained model and its assumptions.")
st.caption("Thermofluid Lab • Experiment 1 • Cylinder flow")
