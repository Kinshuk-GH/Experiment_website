FLOW AND THERMAL EXPLORATORY — LOCAL TEST PACKAGE

INSTALL AND RUN
1. Back up your existing project. Extract ALL files from this ZIP into one folder.
2. Use your original training/app environment if possible. Install UI requirements:
   python -m pip install -r requirements.txt
3. Install the SAME Scimba Torch dependency/version used to train your checkpoint.
   It is not pinned here because the supplied files did not identify its distribution/version.
4. Place cylinder_air_deeponet.pt beside app.py and cylinder_model.py.
5. Run: streamlit run app.py
6. Open the URL printed by Streamlit (normally http://localhost:8501).

WITHOUT THE CHECKPOINT
The homepage, experiment description, C_d table, teaching diagrams, and their downloads
work independently of the trained model. Run experiment reports the missing dependency
or checkpoint without hiding these tools. No fabricated flow results are substituted.

CHECKPOINT SECURITY
Only load your own trusted checkpoint. Loading now uses weights_only=True for safety.
If an older checkpoint contains custom Python objects and fails, export a plain state_dict
from your trusted training environment rather than blindly enabling unsafe unpickling.
Original architecture and scaling are retained in cylinder_model.py.

OPTIONAL GEMMA (NOT REQUIRED FOR BASIC TESTING)
Install Ollama separately, accept/review the exact Gemma model terms, and obtain a model
supported by your machine. For example, with an Ollama installation supporting this tag:
   ollama pull gemma3:4b
Run the Ollama server if it is not already running (ollama serve).
Before starting Streamlit set the installed model name:
  Windows PowerShell: $env:OLLAMA_MODEL="gemma3:4b"
  macOS/Linux: export OLLAMA_MODEL="gemma3:4b"
Then run streamlit run app.py. AI is disabled unless OLLAMA_MODEL is set.
The app sends only a numerical summary to http://127.0.0.1:11434/api/generate.
The Ollama service must run on the same machine as the Streamlit backend, not just
on a student's browser machine. This package does not download/install a model.
Model hardware capacity, integration, and explanations must be tested locally.
This is a prototype, not a commercial licence/privacy/security clearance.

OPTIONAL STATIC HTML
index-2.html is an alternative landing page, not necessary for the Streamlit app.
Its experiment link uses localhost:8501 for local tests. Replace this origin with your
actual deployed Streamlit address when publishing. Retain /?page=cylinder.

TEST CHECKLIST
- Homepage: title/subtitle, quiet background, two experiment categories.
- Open experiment and return to Main page.
- Enter Re = 1, 5, 10 and 7.25; try out-of-range and nonnumeric input.
- Click Run experiment. Compare displayed run label and controls.
- Download PNG/PDF from each flow tab.
- Open C_d tool; enter two or more rows, add/delete rows, plot and download PNG/PDF/CSV.
- Check missing values, duplicate Re, negative Cd and range limits.
- Edit the table, resubmit and verify graph changes. No drag calculation is performed.
- Download boundary and inlet-speed teaching diagrams.
- If configured, test Explain my results; stop Ollama to test graceful failure.
- Model predictions are kept in session state, so table actions do not rerun inference.

Subtitle retained from the prior proposal: Discover Engineering Through Virtual Experiments.
Additional suggested pressure-coefficient, vorticity and drag-integration plots are not
implemented; these require separate scientific validation. No automatic Cd is claimed.
