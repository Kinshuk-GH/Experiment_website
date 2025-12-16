import streamlit as st

# Import your orificemeter module (same folder as app.py)
import Orificemeter  # make sure file is named Orificemeter.py

st.set_page_config(page_title="Thermofluid Lab", layout="centered")

# ---- Homepage ----
st.title("Thermofluid Lab")

st.header("Experiements")  # as requested (typo preserved)

st.write(
    "Select an experiment from the list below to run the corresponding Python code "
    "and view the results."
)

# Show Orificemeter as an experiment entry
st.subheader("Available Experiments")
if st.button("Orificemeter"):
    st.markdown("### Orificemeter Results")

    # Option 1: Call a function defined inside Orificemeter.py
    # You define this function in Orificemeter.py, and it should return a string,
    # dictionary, dataframe, etc. to display.
    if hasattr(Orificemeter, "run_orificemeter"):
        results = Orificemeter.run_orificemeter()
        st.write(results)
    elif hasattr(Orificemeter, "main"):
        # Fallback: if you have a main() function
        results = Orificemeter.main()
        st.write(results)
    else:
        st.error(
            "Orificemeter.py does not define a `run_orificemeter()` or `main()` "
            "function to call."
        )

else:
    st.info("Click the **Orificemeter** button above to run the experiment.")
