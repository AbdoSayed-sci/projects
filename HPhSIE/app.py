import streamlit as st
import numpy as np
from scipy.optimize import fsolve
from groq import Groq
# ==========================================
# integrate the chatbot
# ==========================================
# 1. Safely pull the key from Streamlit's secrets vault
try:
    api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Missing Groq API Key! Please configure GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

# 2. Initialize the Groq client with the secret key
client = Groq(api_key=api_key)

# 3. Example of calling Llama 3 in your calculation function
def get_ai_insight(eq_name, inputs, result):
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Fast and excellent for physics explanations
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Health Biophysicist peer coaching a student."
                },
                {
                    "role": "user",
                    "content": f"The student used the {eq_name} equation with inputs {inputs} and calculated {result}. In two brief sentences, give a physical or biological reality check of this result."
                }
            ],
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Could not generate AI insight: {e}"
# ==========================================
# 1. VARIABLE REGISTRY
# ==========================================
REGISTRY = {
    "A": {"name": "Final Activity", "unit": "MBq"},
    "A0": {"name": "Initial Activity", "unit": "MBq"},
    "lam": {"name": "Decay Constant", "unit": "s^-1"},
    "t": {"name": "Time", "unit": "s"},
    "I": {"name": "Final Intensity", "unit": "W/m^2"},
    "I0": {"name": "Initial Intensity", "unit": "W/m^2"},
    "mu": {"name": "Linear Atten. Coeff.", "unit": "cm^-1"},
    "x": {"name": "Thickness", "unit": "cm"},
    "I1": {"name": "Intensity 1", "unit": "W/m^2"},
    "d1": {"name": "Distance 1", "unit": "m"},
    "I2": {"name": "Intensity 2", "unit": "W/m^2"},
    "d2": {"name": "Distance 2", "unit": "m"},
    "HVL": {"name": "Half-Value Layer", "unit": "cm"},
    "Teff": {"name": "Effective Half-Life", "unit": "s"},
    "Tp": {"name": "Physical Half-Life", "unit": "s"},
    "Tb": {"name": "Biological Half-Life", "unit": "s"},
    "X": {"name": "Exposure", "unit": "C/kg"},
    "Q": {"name": "Charge", "unit": "C"},
    "m": {"name": "Mass", "unit": "kg"},
    "D": {"name": "Absorbed Dose", "unit": "Gy"},
    "E": {"name": "Energy Deposited", "unit": "J"},
    "X_dot": {"name": "Exposure Rate", "unit": "R/h"},
    "Gamma": {"name": "Specific Gamma Ray Const.", "unit": "R·cm^2/mCi·h"},

}

# ==========================================
# 2. EQUATION LIBRARY
# ==========================================
# Residual functions are designed such that f(vars) = 0
EQUATIONS = [
    {
        "id": "decay",
        "name": "Radioactive Decay Law",
        "latex": r"A = A_0 e^{-\lambda t}",
        "vars": {"A", "A0", "lam", "t"},
        "func": lambda v: v['A'] - v['A0'] * np.exp(-v['lam'] * v['t'])
    },
    {
        "id": "attenuation",
        "name": "Photon Attenuation",
        "latex": r"I = I_0 e^{-\mu x}",
        "vars": {"I", "I0", "mu", "x"},
        "func": lambda v: v['I'] - v['I0'] * np.exp(-v['mu'] * v['x'])
    },
    {
        "id": "inv_square",
        "name": "Inverse Square Law",
        "latex": r"I_1 d_1^2 = I_2 d_2^2",
        "vars": {"I1", "d1", "I2", "d2"},
        "func": lambda v: (v['I1'] * v['d1']**2) - (v['I2'] * v['d2']**2)
    },
    {
        "id": "hvl",
        "name": "Half-Value Layer",
        "latex": r"HVL = \frac{\ln(2)}{\mu}",
        "vars": {"HVL", "mu"},
        "func": lambda v: v['HVL'] - (np.log(2) / v['mu'])
    },
    {
        "id": "eff_half_life",
        "name": "Effective Half-Life",
        "latex": r"\frac{1}{T_{eff}} = \frac{1}{T_p} + \frac{1}{T_b}",
        "vars": {"Teff", "Tp", "Tb"},
        "func": lambda v: (1/v['Teff']) - (1/v['Tp'] + 1/v['Tb'])
    },
    {
        "id": "exposure",
        "name": "Radiation Exposure",
        "latex": r"X = \frac{Q}{m}",
        "vars": {"X", "Q", "m"},
        "func": lambda v: v['X'] - (v['Q'] / v['m'])
    },
    {
        "id": "absorbed_dose",
        "name": "Absorbed Dose",
        "latex": r"D = \frac{E}{m}",
        "vars": {"D", "E", "m"},
        "func": lambda v: v['D'] - (v['E'] / v['m'])
    },
    {
        "id": "exposure_rate",
        "name": "Exposure Rate",
        "latex": r"\dot{X} = \frac{\Gamma A}{d^2}",
        "vars": {"X_dot", "Gamma", "A", "d1"}, # Reused d1 for distance
        "func": lambda v: v['X_dot'] - ((v['Gamma'] * v['A']) / v['d1']**2)
    }
]

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def validate_physics(knowns, unknown_key, result):
    """Physical boundary validations."""
    test_state = knowns.copy()
    test_state[unknown_key] = result
    
    warnings = []
    if any(k in test_state and test_state[k] < 0 for k in ['m', 't', 'd1', 'd2', 'x', 'Tp', 'Tb', 'lam']):
        warnings.append("Warning: Calculated a negative value for a strictly positive physical quantity (mass, time, distance, etc).")
    
    if 'A' in test_state and 'A0' in test_state and test_state['A'] > test_state['A0']:
        warnings.append("Warning: Violation of Thermodynamics - Final activity exceeds initial activity.")
        
    return warnings

def solve_numerical(eq, knowns, unknown_key):
    def objective(x):
        test_vars = knowns.copy()
        # Use abs(x) to keep the solver in physical bounds (no negative mass/time)
        test_vars[unknown_key] = abs(x[0]) 
        return eq['func'](test_vars)

    # List of orders of magnitude to try as a starting point
    # This prevents getting stuck at 1.0
    trial_guesses = [1e-9, 1e-6, 1e-3, 0.1, 1.0, 10.0, 100.0, 1e4, 1e7]
    
    best_guess = 1.0
    min_error = float('inf')

    # Preliminary "Scan" to find a valid starting neighborhood
    for guess in trial_guesses:
        try:
            error = abs(objective([guess]))
            if error < min_error:
                min_error = error
                best_guess = guess
        except:
            continue

    # Final Solve with the best starting point found
    result, info, ier, msg = fsolve(objective, [best_guess], full_output=True)
    
    if ier != 1:
        # Fallback: if fsolve fails, return the best we found during the scan
        return best_guess
        
    return abs(result[0])

# ==========================================
# 4. STREAMLIT UI & INFERENCE ENGINE
# ==========================================
st.set_page_config(page_title="Biophysics Inference Engine", layout="centered", page_icon="⚛️")

# Custom Dark Theme CSS
st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #E0E0E0; }
    .stSelectbox label, .stTextInput label { color: #A0A0A0; font-weight: bold; }
    h1, h2, h3 { color: #4DA8DA; }
    .stAlert { background-color: #1E1E1E; border: 1px solid #333; }
    </style>
""", unsafe_allow_html=True)

st.title("⚛️ Socratic Inference Engine")
st.markdown("*Health Physics & Biophysics Numerical Solver*")
st.divider()

if "row_count" not in st.session_state:
    st.session_state.row_count = 3

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("➕ Add Variable Row"):
        st.session_state.row_count += 1
with col2:
    if st.button("➖ Remove Row") and st.session_state.row_count > 1:
        st.session_state.row_count -= 1

st.markdown("### Input State")
st.info("Select variables and input values. **Leave exactly ONE value blank** to act as the unknown.")

# Dynamic Inputs
user_inputs = {}
for i in range(st.session_state.row_count):
    c1, c2 = st.columns([2, 3])
    with c1:
        var_choice = st.selectbox(
            f"Variable {i+1}", 
            options=["None"] + list(REGISTRY.keys()), 
            format_func=lambda x: f"{x} - {REGISTRY[x]['name']}" if x != "None" else "Select...",
            key=f"var_{i}"
        )
    with c2:
        val_input = st.text_input(
            "Value (Leave blank to solve)", 
            key=f"val_{i}", 
            placeholder=f"Unit: {REGISTRY[var_choice]['unit'] if var_choice != 'None' else ''}"
        )
        
    if var_choice != "None":
        user_inputs[var_choice] = val_input

st.divider()

if st.button("🔬 Analyze & Solve", type="primary"):
    selected_vars = set(user_inputs.keys())
    
    if not selected_vars:
        st.warning("Please input at least one variable.")
        st.stop()
        
    # Categorize Knowns and Unknowns
    knowns = {}
    unknowns = []
    
    for k, v in user_inputs.items():
        if v.strip() == "":
            unknowns.append(k)
        else:
            try:
                knowns[k] = float(v)
            except ValueError:
                st.error(f"Invalid numerical input for {k}.")
                st.stop()
                
    if len(unknowns) != 1:
        st.error(f"Engine Exception: You must leave exactly ONE variable blank. Currently missing: {len(unknowns)}.")
        st.stop()
        
    unknown_key = unknowns[0]
    
    # --- INFERENCE ENGINE LOGIC ---
    matches = []
    for eq in EQUATIONS:
        # A match occurs if the provided variables EXACTLY match the equation's required variables
        if selected_vars == eq['vars']:
            matches.append(eq)
            
    # --- SOCRATIC PEER LAYER ---
    if len(matches) == 0:
        st.error("### ❌ Zero Matches Found")
        st.write("The provided variable set does not map perfectly to a known physical law in the registry.")
        
        # Socratic Suggestion: Find closest matches
        st.markdown("#### Did you mean to use one of these?")
        for eq in EQUATIONS:
            missing = eq['vars'] - selected_vars
            extra = selected_vars - eq['vars']
            if len(missing) <= 2 and len(extra) == 0:
                st.info(f"**{eq['name']}**: You are missing variables: `{', '.join(missing)}`")
                st.latex(eq['latex'])
                
    elif len(matches) > 1:
        # Collision detection (Unlikely with exact set matching, but crucial for robust physics engines)
        st.warning("### ⚠️ Physical Context Required")
        st.write("Multiple laws match this exact variable signature. Please clarify the physical scenario.")
        context = st.radio("Select Intended Phenomenon:", [m['name'] for m in matches])
        
        # In a full flow, this would trigger a session state update. For immediate feedback:
        selected_eq = next(m for m in matches if m['name'] == context)
        st.latex(selected_eq['latex'])
        # Solve would proceed here based on selection.
        
    elif len(matches) == 1:
        eq = matches[0]
        st.success(f"### ✅ Identified: {eq['name']}")
        st.latex(eq['latex'])
        
        try:
            result = solve_numerical(eq, knowns, unknown_key)
            
            # Validation Layer
            warnings = validate_physics(knowns, unknown_key, result)
            for w in warnings:
                st.warning(w)
                
            st.markdown(f"### **Solved Result:**")
            st.markdown(f"**{REGISTRY[unknown_key]['name']} ({unknown_key})** = `{result:.4g}` {REGISTRY[unknown_key]['unit']}")
            
        except Exception as e:
            st.error(f"Numerical Solver Failed: {e}")
# Create a designated slot in session state for the insight
if "ai_insight" not in st.session_state:
    st.session_state.ai_insight = ""

if st.button("🔬 Analyze & Solve", type="primary"):
    # ... your existing solver logic ...
    # calculation_result = result
    
    # 1. Call Groq and save it straight to session state
    with st.spinner("🧠 Consulting the Socratic Peer..."):
        st.session_state.ai_insight = get_ai_insight(
            eq['name'], 
            knowns, 
            f"{result:.4g} {REGISTRY[unknown_key]['unit']}"
        )

# 2. Render the sentence OUTSIDE the button click loop
if st.session_state.ai_insight:
    st.markdown("---")
    st.markdown("### 💬 Socratic Peer Insight")
    st.info(st.session_state.ai_insight)
