# Thermofluid Lab — Cylinder Flow

A Streamlit app for a student-friendly virtual experiment: air flowing past a circular cylinder. It loads an already-trained PyTorch checkpoint and does **not** train the network at app startup.

## Files

- `app.py` — interactive experiment and model inference.
- `index.html` — static landing page; replace its example app URL after deploying Streamlit.
- `requirements.txt` — base Python dependencies.
- `cylinder_air_deeponet.pt` — **you must copy this checkpoint into this folder**. It is expected to be created by running the notebook's final export cell.

## Prepare the checkpoint

Run `Cylinder_Flow_DeepONet_LR_Decay.ipynb` through its final export cell. The notebook writes:

`cylinder_air_lr_decay_results/cylinder_air_deeponet.pt`

Copy that file beside `app.py` and rename it to `cylinder_air_deeponet.pt` if needed. The checkpoint contains the state dictionary and model metadata. Do not include the full training loop in the web app.

The app uses the same `CylinderDeepONet` architecture, coordinate scaling, air properties, and hard boundary-condition construction as the notebook. It also requires the same Scimba Torch `GenericMLP` implementation that was used during training. Install a compatible version in the deployment environment.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. If `scimba_torch` is unavailable, install the same Scimba Torch package/environment used by the notebook first. Do not assume the unrelated `scrimba` package is equivalent.

## Deploy / use with Kaggle

Kaggle notebooks are useful for running the notebook and storing/exporting the checkpoint, but a notebook is not by itself a permanent public website host. To share a live interactive Streamlit site, deploy this folder (including the `.pt` checkpoint) to a Streamlit-compatible host, or use a temporary tunnel from a running Kaggle session for a short demo. Temporary tunnels are not permanent and should not be treated as production hosting.

The `index.html` file is a static landing page. It cannot directly run Python or load a PyTorch checkpoint in the browser; its experiment button must link to the deployed Streamlit app. Replace `https://YOUR-STREAMLIT-APP-URL.streamlit.app` with the real URL.

## Add more experiments later

1. Put each new experiment in a separate Python module, e.g. `pipe_flow.py`.
2. Add a new experiment card to `index.html`.
3. Add navigation/selection in `app.py` (or make a separate Streamlit page in a `pages/` folder).
4. For each trained model, load its checkpoint with `torch.load(..., map_location=device)` and instantiate the matching architecture. Keep training code out of the deployed app.

## Current experiment's assumptions

- Steady, two-dimensional air flow around a stationary circular cylinder.
- Reynolds number supported by the trained model: 1 to 10.
- Approximate dry-air properties at 20 °C.
- Assumed cylinder diameter: 0.01 m. Edit `D_M` in `app.py` only if you want dimensional speed/pressure values for a different physical diameter; the dimensionless prediction is unchanged.
- Predicted fields are model outputs and should be validated against experimental/reference data before being treated as quantitatively accurate.
