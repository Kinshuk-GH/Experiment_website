from pathlib import Path
import os
import json
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from lab_utils import valid_re, validate_table, figure_bytes

TITLE = "Flow and Thermal Exploratory"
SUBTITLE = "Discover Engineering Through Virtual Experiments"
EXPERIMENT = "Flow across cylinder with Re = 1 to 10"
st.set_page_config(page_title=TITLE, page_icon="🌊", layout="wide")
st.markdown("""<style>
.stApp {background:#F7F9FC;color:#16324F;}
.hero {background:linear-gradient(110deg,#FFFFFF,#EEF5FB);padding:2.2rem;
border:1px solid #DFE7F0;border-radius:14px;margin-bottom:1.5rem;}
.hero h1 {color:#16324F;margin:0 0 .6rem;}
.hero p {color:#43566B;margin:0;}
div[data-testid="stVerticalBlockBorderWrapper"] {background:white;border-radius:12px;}
</style>""", unsafe_allow_html=True)

def show_plot(fig, name):
    st.pyplot(fig)
    a, b = st.columns(2)
    a.download_button("↓ Download PNG", figure_bytes(fig,"png"), name+".png",
                      "image/png", key=name+"_png", on_click="ignore")
    b.download_button("↓ Download PDF", figure_bytes(fig,"pdf"), name+".pdf",
                      "application/pdf", key=name+"_pdf", on_click="ignore")
    plt.close(fig)

if st.query_params.get("page", "home") != "cylinder":
    st.markdown(f'<div class="hero"><h1>{TITLE}</h1><p>{SUBTITLE}</p></div>', unsafe_allow_html=True)
    st.write("Welcome to an interactive virtual laboratory for fluid mechanics and heat transfer. "
             "Explore physical ideas, vary experiment inputs, and investigate model predictions through plots and guided activities.")
    st.header("List of experiments")
    a,b=st.columns(2)
    with a:
        with st.container(border=True):
            st.subheader("Fluid Mechanics")
            st.markdown(
                f'1. <a href="?page=cylinder" target="_self">{EXPERIMENT}</a>',
                unsafe_allow_html=True,
            )
            st.write("Investigate air speed, pressure, and flow direction around a stationary circular cylinder.")
    with b:
        with st.container(border=True):
            st.subheader("Heat Transfer")
            st.write("Experiments coming soon.")
    st.stop()

st.markdown(
    '<a href="?page=home" target="_self">← Back to Main page</a>',
    unsafe_allow_html=True,
)
st.title(EXPERIMENT)
st.caption(TITLE+" · Fluid Mechanics")
st.write("Imagine a gentle stream of air meeting a round pole. The air must move around it, "
         "changing speed and pressure. In this experiment a saved physics-informed neural network predicts "
         "steady, two-dimensional flow around a stationary cylinder. No training takes place on the website.")
with st.expander("Understand the experiment", expanded=True):
    st.subheader("What is Reynolds number?")
    st.latex(r"Re=\frac{\rho U_\infty D}{\mu}=\frac{U_\infty D}{\nu}")
    st.write("Re compares inertia (the tendency of moving air to keep moving) with viscosity "
             "(internal friction that smooths differences in velocity). Here D = 0.01 m, "
             "air density = 1.204 kg/m³ and dynamic viscosity = 1.825 × 10⁻⁵ Pa·s, "
             "approximating dry air at 20 °C.")
    st.latex(r"U_\infty=\frac{Re\,\nu}{D}")
    st.write("Diameter and air properties are fixed. Increasing Re therefore increases the inlet speed "
             "in direct proportion: doubling Re doubles speed. Re = 1 and 10 correspond to approximately "
             "0.00152 and 0.0152 m/s respectively.")
    st.subheader("What should you look for?")
    st.write("Across Re = 1–10, expect steady laminar flow, symmetric above and below the cylinder. "
             "Near Re = 1, viscosity strongly influences the motion and flow remains attached. "
             "As Re increases, front–back symmetry becomes less pronounced; a small, steady pair of "
             "recirculating regions can develop behind the cylinder towards the upper part of this range. "
             "Alternating vortex shedding is not expected here. These are expected physical trends, "
             "not a guarantee that a particular neural-network prediction resolves every feature.")
    st.subheader("Why these boundary conditions?")
    st.markdown("""- Cylinder: zero velocity (no slip and no penetration), because the solid is stationary.
- Inlet: uniform horizontal flow, representing the undisturbed approaching air.
- Top and bottom: uniform far-field velocity, approximating open flow rather than solid channel walls.
- Outlet: zero gauge pressure fixes the pressure reference. The supplied model does not hard-code an outlet velocity; its downstream behaviour also depends on training.
- Domain: x/D = −20 to 40 and y/D = −20 to 20. Distant edges reduce artificial confinement, but domain independence still requires validation.
""")
    st.caption("Pressure is scaled using μU∞/D, as in the supplied model—not automatically ½ρU∞².")
    st.markdown("Background reading: [low-Re cylinder flow](https://www.wolfdynamics.com/wiki/tut_2D_cylinder.pdf)")

with st.sidebar:
    st.header("Experiment controls")
    with st.form("flow_controls"):
        re_input=st.number_input("Reynolds number (Re)", min_value=1.0,max_value=10.0,
                                value=5.0,step=0.1,format="%.2f")
        region=st.selectbox("Region to view",["Near the cylinder","Larger region"])
        run=st.form_submit_button("Run experiment")
    st.caption("Valid range: 1 ≤ Re ≤ 10. Decimal values are allowed. Click Run experiment to apply changes.")

st.header("Record and analyse your results")
if st.button("Create C_d vs Re Plot"):
    st.session_state["show_cd"]=True
if st.session_state.get("show_cd"):
    st.write("Enter your observations. This tool plots user-entered data; it does not calculate drag from the model.")
    with st.form("cd_form"):
        edited=st.data_editor(pd.DataFrame({"Re":pd.Series([None]*3,dtype="float64"),
                                            "Cd":pd.Series([None]*3,dtype="float64")}),
            num_rows="dynamic",hide_index=True,key="cd_editor",
            column_config={"Re":st.column_config.NumberColumn("Re",min_value=1.,max_value=10.),
                           "Cd":st.column_config.NumberColumn("C_d",min_value=0.)})
        submitted=st.form_submit_button("Plot data")
    if submitted:
        try:
            st.session_state["cd_data"]=validate_table(edited)
        except ValueError as exc:
            st.session_state.pop("cd_data",None)
            st.error(str(exc))
    if "cd_data" in st.session_state:
        data=st.session_state["cd_data"]
        fig,ax=plt.subplots(figsize=(7,4))
        ax.plot(data.Re,data.Cd,"o-",color="#1769AA")
        ax.set(xlabel="Reynolds number, Re",ylabel="Drag coefficient, C_d",
               title="C_d vs Re — user-entered data")
        ax.grid(alpha=.25)
        show_plot(fig,"cylinder_Cd_vs_Re")
        st.caption("This graph shows the last successfully submitted table; submit again after editing.")
        st.download_button("↓ Download data (CSV)",data.to_csv(index=False).encode(),
                           "cylinder_Cd_vs_Re.csv","text/csv",on_click="ignore")

st.header("Teaching diagrams")
with st.expander("Setup and velocity–Re relationship"):
    fig,ax=plt.subplots(figsize=(9,3))
    ax.add_patch(plt.Rectangle((-20,-20),60,40,fill=False,ec="#1769AA"))
    ax.add_patch(plt.Circle((0,0),.5,color="gray"))
    ax.annotate("Stationary cylinder (D)",xy=(0,0),xytext=(5,6),arrowprops={"arrowstyle":"->"})
    for yy in [-12,0,12]:
        ax.arrow(-28,yy,7,0,head_width=1.5,color="#1769AA",length_includes_head=True)
    ax.text(-29,23,"Uniform inlet")
    ax.text(26,23,"Outlet: p = 0")
    ax.text(-9,-29,"Top/bottom: uniform far field; cylinder not to scale")
    ax.set(xlim=(-32,48),ylim=(-33,30),xlabel="x/D",ylabel="y/D",title="Computational boundaries")
    ax.set_aspect("equal")
    show_plot(fig,"boundary_conditions")
    fig,ax=plt.subplots(figsize=(7,3.5))
    re=np.linspace(1,10,100)
    ax.plot(re,re*(1.825e-5/1.204)/.01)
    ax.set(xlabel="Reynolds number, Re",ylabel="Inlet speed (m/s)",title="Fixed diameter and air properties: speed ∝ Re")
    ax.grid(alpha=.25)
    show_plot(fig,"inlet_speed_vs_Re")

if run:
    if not valid_re(re_input):
        st.error("Enter a finite Reynolds number between 1 and 10.")
    else:
        try:
            from cylinder_model import load_model,predict_field,MODEL_PATH
            if not MODEL_PATH.exists():
                raise FileNotFoundError("Place cylinder_air_deeponet.pt beside app.py.")
            model,device,_=load_model(str(MODEL_PATH),MODEL_PATH.stat().st_mtime)
            xr,yr=((-4,12),(-4,4)) if region=="Near the cylinder" else ((-20,40),(-20,20))
            with st.spinner("Calculating the predicted field..."):
                field=predict_field(model,device,re_input,xr,yr)
            fluid_mask = field[0]**2 + field[1]**2 > 0.5**2
            if not all(np.isfinite(a[fluid_mask]).all() for a in field[2:]):
                raise ValueError("The model returned non-finite predictions.")
            st.session_state["flow_result"]={"re":re_input,"region":region,"field":field}
            st.session_state.pop("ai_text",None)
        except Exception as exc:
            st.session_state.pop("flow_result",None)
            st.error(f"Experiment could not run: {exc}")
            st.info("The homepage, teaching diagrams, and C_d plotting tool can still be tested without the checkpoint.")

result=st.session_state.get("flow_result")
if result:
    re=result["re"]
    XX,YY,u,v,speed,pressure=result["field"]
    st.header("Predicted results")
    st.caption(f"Displayed run: Re = {re:g}; {result['region']}. Controls affect results only after Run experiment.")
    a,b,c=st.columns(3)
    a.metric("Reynolds number",f"{re:g}")
    b.metric("Inlet speed",f"{re*(1.825e-5/1.204)/.01:.5f} m/s")
    c.metric("Cylinder diameter","1 cm")
    tabs=st.tabs(["Speed","Pressure","Flow direction"])
    for i,(tab,values,label,title) in enumerate(zip(tabs,[speed,pressure,None],
            ["Speed (m/s)","Gauge pressure (Pa)",""],["Speed","Pressure","Streamlines"])):
        with tab:
            fig,ax=plt.subplots(figsize=(10,4.5))
            if values is not None:
                im=ax.contourf(XX,YY,np.ma.masked_invalid(values),levels=32)
                fig.colorbar(im,ax=ax,label=label)
            else:
                ax.streamplot(XX[0],YY[:,0],np.ma.masked_invalid(u),np.ma.masked_invalid(v),density=1.2)
            ax.add_patch(plt.Circle((0,0),.5,color="lightgray",ec="black",zorder=4))
            ax.set(xlabel="x/D",ylabel="y/D",title=f"Predicted {title.lower()} at Re = {re:g}")
            ax.set_aspect("equal")
            show_plot(fig,f"cylinder_{title.lower()}_Re_{re:g}")
    st.info("These are model predictions, not measurements. Validate accuracy before using them for engineering decisions.")
    st.subheader("Explain my results — optional local Gemma")
    st.caption("Requires a separately running local Ollama server and an installed Gemma model. No paid API is configured.")
    model_name=os.environ.get("OLLAMA_MODEL","")
    if not model_name:
        st.info("AI is disabled. Set OLLAMA_MODEL to your installed Gemma model name, then restart the app. See README.")
    if st.button("Explain my results",disabled=not bool(model_name)):
        summary={"Re":re,"diameter_m":.01,"inlet_speed_m_s":re*(1.825e-5/1.204)/.01,
                 "region":result["region"],"speed_min_m_s":float(np.nanmin(speed)),
                 "speed_max_m_s":float(np.nanmax(speed)),"pressure_min_Pa":float(np.nanmin(pressure)),
                 "pressure_max_Pa":float(np.nanmax(pressure))}
        prompt=("You are a careful fluid-mechanics teacher. Explain the supplied simulation summary simply. "
            "The model is steady 2D air flow past a stationary 1-cm cylinder at Re 1–10. "
            "Diameter and properties are fixed, so inlet speed is proportional to Re. "
            "Wall: no-slip; inlet/top/bottom: uniform far field; outlet gauge pressure zero. "
            "Only extrema are supplied, not their locations. Do not infer wake detection, drag, "
            "separation locations, validation accuracy or vortex shedding from these extrema. "
            "Separate expected physical trends from supplied observations. Never invent Cd. "
            "Do not give engineering safety assurances. Keep under 250 words. Data: "+json.dumps(summary))
        try:
            req=urllib.request.Request("http://127.0.0.1:11434/api/generate",
                data=json.dumps({"model":model_name,"prompt":prompt,"stream":False}).encode(),
                headers={"Content-Type":"application/json"})
            with st.spinner("Generating a local AI explanation..."):
                with urllib.request.urlopen(req,timeout=180) as response:
                    text=json.load(response).get("response","")
            if not text.strip():
                raise ValueError("The model returned an empty explanation.")
            st.session_state["ai_text"]=text
        except Exception as exc:
            st.error(f"Local AI unavailable: {exc}. The experiment remains usable.")
    if st.session_state.get("ai_text"):
        st.warning("AI-generated explanation: may contain errors. Verify against the experiment notes.")
        st.write(st.session_state["ai_text"])
else:
    st.info("Enter Re in the sidebar and click Run experiment to generate flow plots.")

st.subheader("Try an investigation")
st.write("Run Re = 1, 5, and 10. Compare flow direction, pressure, and wake structure. "
         "Download your figures and record one observation and one question.")
st.caption(TITLE+" · Cylinder flow")
