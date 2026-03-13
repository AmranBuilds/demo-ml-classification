import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page Title
st.title("Classification Foundation: The Sigmoid Function")

# Sidebar for variable adjustment
st.sidebar.header("Adjust Mathematical Variables")
w = st.sidebar.slider("Weight (w)", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
b = st.sidebar.slider("Bias (b)", min_value=-10.0, max_value=10.0, value=0.0, step=0.5)
x_val = st.sidebar.slider("Specific Input (x)", min_value=-10.0, max_value=10.0, value=0.0, step=0.5)

# General LaTeX Equation
st.subheader("General Mathematical Formula")
st.latex(r"P(y=1|x) = \frac{1}{1 + e^{-(wx + b)}}")

# Dynamic LaTeX Equation
st.subheader("Dynamic Formula Based on Sliders")
st.latex(f"P(y=1|x={x_val}) = \\frac{{1}}{{1 + e^{{-({w} \\cdot {x_val} + {b})}}}}")

# Calculate probability for dynamic display
z = w * x_val + b
prob = 1 / (1 + np.exp(-z))
st.latex(f"P(y=1) = {prob:.4f}")

# Visual Representation
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

# ELI5 Chart Explanation
st.subheader("How to Read This Chart")
st.markdown("""
* **The Blue Line:** This is the sorting rule. It shows the calculated probability for any given input. 
* **The Gray Dashed Line:** This is the decision boundary. If a point is above this line (Probability > 0.5), the computer sorts it into Class 1. If it is below the line, it sorts it into Class 0.
* **The Red Dot:** This represents your specific item based on the `x` slider. Watch how changing the weight (`w`) changes the steepness of the rule, and changing the bias (`b`) shifts the rule left or right.
""")
