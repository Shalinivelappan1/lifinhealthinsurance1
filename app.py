import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Page Config
# =====================================================
st.set_page_config(page_title="LiFin Health Insurance Lab", layout="wide")

st.title("🏥 LiFin Health Insurance Lab — Developed by Prof.Shalini Velappan, IIM Trichy")

st.caption("""
This is a **teaching simulator** to understand:

• Deductibles, co-pay, sub-limits  
• Room rent proportional deduction (Indian reality)  
• Family floater vs separate policies  
• Super top-up layers  
• Risk distribution & tail risk over 30 years  

It is **not** an actuarial pricing engine.
""")

# =====================================================
# Sidebar: Family Structure
# =====================================================
st.sidebar.header("👨‍👩‍👧‍👦 Family Structure")

include_spouse = st.sidebar.checkbox("Include Spouse", value=True)
num_children = st.sidebar.selectbox("Number of Children", [0, 1, 2], index=1)
num_parents = st.sidebar.selectbox("Number of Parents", [0, 1, 2], index=2)

members = ["Self"]
if include_spouse:
    members.append("Spouse")
for i in range(num_children):
    members.append(f"Child {i+1}")
for i in range(num_parents):
    members.append(f"Parent {i+1}")

st.sidebar.markdown(f"**Total Covered Members:** {len(members)}")

# =====================================================
# Sidebar: Medical Environment
# =====================================================
st.sidebar.header("📈 Medical Environment")

inflation = st.sidebar.slider("Medical Inflation (%)", 4.0, 15.0, 8.0) / 100
years = 30

# Base cost ranges
normal_min = st.sidebar.number_input("Normal Event Cost - Min (₹)", value=50_000, step=10_000)
normal_max = st.sidebar.number_input("Normal Event Cost - Max (₹)", value=300_000, step=50_000)

major_min = st.sidebar.number_input("Major Event Cost - Min (₹)", value=500_000, step=100_000)
major_max = st.sidebar.number_input("Major Event Cost - Max (₹)", value=2_500_000, step=100_000)

# =====================================================
# Sidebar: Risk Parameters
# =====================================================
st.sidebar.header("⚠️ Annual Risk Parameters")

p_normal_base = st.sidebar.slider("Prob. Normal Hospitalization (%) (Non-parents)", 0.0, 20.0, 5.0) / 100
p_major_base = st.sidebar.slider("Prob. Major Illness (%) (Non-parents)", 0.0, 10.0, 1.0) / 100

st.sidebar.markdown("**Parents Risk (User Controlled)**")
p_normal_parent = st.sidebar.slider("Parent: Prob. Normal Hospitalization (%)", 0.0, 30.0, 10.0) / 100
p_major_parent = st.sidebar.slider("Parent: Prob. Major Illness (%)", 0.0, 20.0, 3.0) / 100

parent_cost_multiplier = st.sidebar.slider("Parent Cost Multiplier", 1.0, 3.0, 1.5)

# =====================================================
# Policy Input UI
# =====================================================
def policy_ui(label, default_cover, default_premium):
    st.sidebar.header(f"📜 {label}")

    floater = st.sidebar.checkbox(f"{label}: Family Floater", value=True)

    cover = st.sidebar.number_input(f"{label}: Base Cover (₹)", value=default_cover, step=100_000)
    deductible = st.sidebar.number_input(f"{label}: Deductible (₹)", value=100_000, step=50_000)
    copay = st.sidebar.slider(f"{label}: Co-pay (%)", 0, 50, 10) / 100

    room_limit = st.sidebar.number_input(f"{label}: Room Rent Limit (₹ per day)", value=5_000, step=1_000)

    premium = st.sidebar.number_input(f"{label}: Annual Premium (₹)", value=default_premium, step=2_000)

    st.sidebar.markdown(f"**{label}: Super Top-Up Layer**")
    has_topup = st.sidebar.checkbox(f"{label}: Enable Super Top-Up", value=False)
    topup_cover = st.sidebar.number_input(f"{label}: Top-Up Cover (₹)", value=2_000_000, step=500_000)
    topup_threshold = st.sidebar.number_input(f"{label}: Top-Up Threshold (₹)", value=default_cover, step=100_000)
    topup_premium = st.sidebar.number_input(f"{label}: Top-Up Premium (₹)", value=8_000, step=1_000)

    return {
        "floater": floater,
        "cover": cover,
        "deductible": deductible,
        "copay": copay,
        "room_limit": room_limit,
        "premium": premium,
        "has_topup": has_topup,
        "topup_cover": topup_cover,
        "topup_threshold": topup_threshold,
        "topup_premium": topup_premium
    }

policyA = policy_ui("Policy A", default_cover=1_000_000, default_premium=25_000)
policyB = policy_ui("Policy B", default_cover=2_000_000, default_premium=45_000)

# =====================================================
# Teaching Explanation
# =====================================================
with st.expander("ℹ️ How claims are processed in this simulator"):
    st.markdown("""
Order of logic:

1) A medical bill is generated  
2) Room rent limit may cause **proportional deduction** of entire bill  
3) Deductible is applied  
4) Co-pay is applied  
5) Base policy cover is applied  
6) If enabled, **Super top-up** kicks in after threshold  
7) Remainder = Out-of-pocket  

This reflects **real Indian policy mechanics** approximately.
""")

# =====================================================
# Simulation Engine
# =====================================================
def simulate_year_for_member(is_parent):
    if is_parent:
        pN = p_normal_parent
        pM = p_major_parent
        cost_mult = parent_cost_multiplier
    else:
        pN = p_normal_base
        pM = p_major_base
        cost_mult = 1.0

    u = np.random.rand()

    if u < pM:
        cost = np.random.uniform(major_min, major_max) * cost_mult
    elif u < pM + pN:
        cost = np.random.uniform(normal_min, normal_max) * cost_mult
    else:
        cost = 0

    # Room choice simulation (simple)
    if cost > 0:
        room_cost = np.random.choice([3000, 5000, 8000, 12000])
    else:
        room_cost = 0

    return cost, room_cost

def apply_policy(policy, total_claim, avg_room_cost):
    if total_claim <= 0:
        return 0

    # Room rent proportional deduction
    if avg_room_cost > policy["room_limit"]:
        ratio = policy["room_limit"] / avg_room_cost
    else:
        ratio = 1.0

    admissible = total_claim * ratio

    # Deductible
    remaining = max(admissible - policy["deductible"], 0)

    # Co-pay
    insurer_share = remaining * (1 - policy["copay"])

    # Base cover
    paid_by_base = min(insurer_share, policy["cover"])

    leftover = insurer_share - paid_by_base

    # Super top-up
    paid_by_topup = 0
    if policy["has_topup"] and admissible > policy["topup_threshold"]:
        paid_by_topup = min(leftover, policy["topup_cover"])

    total_paid = paid_by_base + paid_by_topup

    oop = total_claim - total_paid
    return oop

def simulate_lifetime(policy):
    total_oop = 0

    for t in range(years):
        infl = (1 + inflation) ** t

        yearly_claim = 0
        rooms = []

        for m in members:
            is_parent = "Parent" in m
            cost, room = simulate_year_for_member(is_parent)
            yearly_claim += cost * infl
            if cost > 0:
                rooms.append(room)

        if yearly_claim == 0:
            continue

        avg_room = np.mean(rooms) if len(rooms) > 0 else 0

        oop = apply_policy(policy, yearly_claim, avg_room)
        total_oop += oop

    total_oop += policy["premium"] * years
    if policy["has_topup"]:
        total_oop += policy["topup_premium"] * years

    return total_oop

# =====================================================
# Run Simulation
# =====================================================
st.markdown("## 🎲 Monte Carlo Simulation")

sims = st.slider("Number of Simulation Paths", 500, 5000, 2000, step=500)

if st.button("▶️ Run Simulation"):

    no_insurance = []
    polA = []
    polB = []

    for _ in range(sims):
        # No insurance
        total_no = 0
        for t in range(years):
            infl = (1 + inflation) ** t
            yearly = 0
            for m in members:
                is_parent = "Parent" in m
                cost, _ = simulate_year_for_member(is_parent)
                yearly += cost * infl
            total_no += yearly

        no_insurance.append(total_no)
        polA.append(simulate_lifetime(policyA))
        polB.append(simulate_lifetime(policyB))

    no_insurance = np.array(no_insurance)
    polA = np.array(polA)
    polB = np.array(polB)

    # =====================================================
    # Summary Stats
    # =====================================================
    def stats(x):
        return {
            "Mean": np.mean(x),
            "Median": np.median(x),
            "95%ile": np.percentile(x, 95),
            "99%ile": np.percentile(x, 99),
            "Worst": np.max(x)
        }

    s0 = stats(no_insurance)
    sA = stats(polA)
    sB = stats(polB)

    st.markdown("## 📊 Risk Summary (30-year Lifetime Cost)")

    st.write("### No Insurance", {k: f"₹ {v:,.0f}" for k, v in s0.items()})
    st.write("### Policy A", {k: f"₹ {v:,.0f}" for k, v in sA.items()})
    st.write("### Policy B", {k: f"₹ {v:,.0f}" for k, v in sB.items()})

    # =====================================================
    # Distribution Plot
    # =====================================================
    st.markdown("## 📈 Distribution of Lifetime Cost")

    fig, ax = plt.subplots()
    ax.hist(no_insurance, bins=40, alpha=0.5, label="No Insurance")
    ax.hist(polA, bins=40, alpha=0.5, label="Policy A")
    ax.hist(polB, bins=40, alpha=0.5, label="Policy B")
    ax.legend()
    ax.set_xlabel("30-year Total Cost (₹)")
    ax.set_ylabel("Frequency")
    ax.set_title("Lifetime Cost Distribution Comparison")
    st.pyplot(fig)

    # =====================================================
    # Teaching Message
    # =====================================================
    st.success("""
🧠 **Key Lesson**

Good insurance does **not** minimize the average.

It **destroys the tail of the distribution** and protects you from financial ruin.

Always evaluate health insurance by:
• 95th / 99th percentile outcomes  
• Not by the mean.
""")

    # =====================================================
    # Reflection Questions
    # =====================================================
    st.markdown("## 📝 Reflection Questions")

    st.markdown("""
1) Which policy **reduces the worst-case outcomes** more effectively?  
2) How do **room rent limits** distort the meaning of “₹X cover”?  
3) Why is a **super top-up** economically efficient?  
4) When does a **family floater** become dangerous?  
5) How does changing **parent risk** change the decision?
""")

# =====================================================
# Footer
# =====================================================
st.markdown("""
---
⚠️ This is a **teaching simulator**, not a pricing or underwriting engine.
""")
