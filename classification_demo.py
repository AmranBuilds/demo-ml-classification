import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Classification Foundation: Binary Logistic Regression")

# ==========================================
# PART 1: THEORETICAL MECHANICS
# ==========================================
st.header("Part 1: Theoretical Mechanics (The Sigmoid Function)")

# Sidebar - Part 1
st.sidebar.header("Part 1: Abstract Variables")
w = st.sidebar.slider("Weight (w)", min_value=-5.0, max_value=5.0, value=1.0, step=0.1, key="w_slider")
b = st.sidebar.slider("Bias (b)", min_value=-10.0, max_value=10.0, value=0.0, step=0.5, key="b_slider")
x_val = st.sidebar.slider("Specific Input (x)", min_value=-10.0, max_value=10.0, value=0.0, step=0.5, key="x_slider")

# Equations
st.subheader("General Mathematical Formula")
st.latex(r"P(y=1|x) = \frac{1}{1 + e^{-(wx + b)}}")

st.subheader("Dynamic Formula Based on Sliders")
st.latex(f"P(y=1|x={x_val}) = \\frac{{1}}{{1 + e^{{-({w} \\cdot {x_val} + {b})}}}}")

# Calculate pure probability
z = w * x_val + b
prob = 1 / (1 + np.exp(-z))
st.latex(f"P(y=1) = {prob:.4f}")

# Visual Representation - Part 1
st.subheader("Visual Representation")
x_range = np.linspace(-10, 10, 200)
y_range = 1 / (1 + np.exp(-(w * x_range + b)))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x_range, y_range, label="Probability Curve", color='blue')
ax.scatter([x_val], [prob], color='red', zorder=5, s=100, label=f"Selected x ({x_val}, {prob:.2f})")
ax.axhline(0.5, color='gray', linestyle='--', label="Decision Boundary (0.5)")
ax.set_xlabel("Input Feature (x)")
ax.set_ylabel("Probability (P)")
ax.set_ylim(-0.1, 1.1)
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.markdown("""
* **The Blue Line:** Calculated probability for any given input. 
* **The Gray Dashed Line:** Decision boundary. Points above 0.5 are Class 1; points below are Class 0.
* **The Red Dot:** Your specific item based on the `x` slider.
""")

st.markdown("---")

# ==========================================
# PART 2: REAL-WORLD APPLICATION
# ==========================================
st.header("Part 2: Applied Scenario - Water Pump Maintenance")
st.markdown("Predicting if a water hand pump needs immediate maintenance to prevent a community from losing clean water access.")

# Sidebar - Part 2
st.sidebar.header("Part 2: Pump Features")
pump_age = st.sidebar.slider("Pump Age (Years)", min_value=0.0, max_value=20.0, value=5.0, step=0.5, key="age_slider")

# Applied Math (Hidden fixed weights mapping age to failure probability)
w_pump = 0.6  # Rate at which aging increases failure risk
b_pump = -6.0 # Baseline risk (negative means low risk at 0 years)

z_pump = w_pump * pump_age + b_pump
prob_failure = 1 / (1 + np.exp(-z_pump))

st.subheader("Probability of Imminent Failure")
st.latex(f"P(\\text{{Failure}}|\\text{{Age}}={pump_age}) = {prob_failure:.4f}")

# Visual Representation - Part 2
fig2, ax2 = plt.subplots(figsize=(8, 4))
bar_color = 'red' if prob_failure >= 0.5 else 'green'
ax2.bar(["Target Water Pump"], [prob_failure], color=bar_color, width=0.4)
ax2.axhline(0.5, color='gray', linestyle='--', label="Dispatch Mechanic Threshold (0.5)")
ax2.set_ylim(0, 1.1)
ax2.set_ylabel("Probability of Failure")
ax2.legend()
st.pyplot(fig2)

# Action Output
if prob_failure >= 0.5:
    st.error(f"**Action Required:** Failure probability is {prob_failure*100:.1f}%. Dispatching mechanic.")
else:
    st.success(f"**Status Normal:** Failure probability is {prob_failure*100:.1f}%. No action needed.")

st.markdown("""
* **The Bar:** Shows the calculated risk of the specific pump failing based on its age.
* **The Threshold:** If the bar crosses 0.5, the system automatically flags the pump for repair before it breaks.
""")
