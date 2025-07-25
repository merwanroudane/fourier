import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy import optimize
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fourier Unit Root Tests for Structural Breaks",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 5px solid #ff7f0e;
        padding-left: 1rem;
    }
    .theory-box {
        background-color: #f0f8ff;
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .equation-box {
        background-color: #fff5ee;
        border: 2px solid #ff7f0e;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .comparison-box {
        background-color: #f5f5f5;
        border: 2px solid #2ca02c;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📊 Fourier Unit Root Tests for Structural Breaks</h1>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.selectbox(
    "Choose Section:",
    ["Theory & Introduction", "Traditional vs Fourier Methods", "Fourier Unit Root Tests",
     "Interactive Simulations", "Smooth vs Sharp Breaks", "Comparative Analysis"]
)

if section == "Theory & Introduction":
    st.markdown('<div class="section-header">🎯 Theoretical Foundation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-box">
    <h3>Introduction to Fourier-Based Unit Root Tests</h3>
    <p>Traditional unit root tests often fail to distinguish between unit roots and structural breaks, leading to incorrect conclusions about the stationarity of time series. The new generation of Fourier-based unit root tests addresses this issue by using trigonometric functions to capture structural breaks without requiring prior knowledge of break dates.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔍 Key Advantages of Fourier Methods")
        st.markdown("""
        - **No Prior Knowledge Required**: Break dates don't need to be known
        - **Multiple Breaks**: Can handle multiple structural breaks
        - **Smooth Transitions**: Captures gradual changes in parameters
        - **Flexible Forms**: Accommodates various break patterns
        - **Better Power**: Higher statistical power compared to traditional tests
        """)

    with col2:
        st.markdown("### 📈 Mathematical Foundation")
        st.markdown("""
        The Fourier approach uses trigonometric functions to approximate deterministic components:
        """)
        st.latex(r'''
        f(t) = \sum_{k=1}^{n} \left[ a_k \cos\left(\frac{2\pi kt}{T}\right) + b_k \sin\left(\frac{2\pi kt}{T}\right) \right]
        ''')
        st.markdown("where T is the sample size and n is the number of frequencies.")

    st.markdown("""
    <div class="equation-box">
    <h3>Fourier Series Representation</h3>
    <p>Any deterministic function can be approximated using Fourier series:</p>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r'''
    g(t) = \alpha_0 + \sum_{k=1}^{\infty} \left[ \alpha_k \cos\left(\frac{2\pi kt}{T}\right) + \beta_k \sin\left(\frac{2\pi kt}{T}\right) \right]
    ''')

    st.markdown("### 🌊 Visualization of Fourier Components")

    # Interactive Fourier visualization
    freq_slider = st.slider("Number of Fourier Components", 1, 10, 3)
    T = 100
    t = np.linspace(0, T, T)

    fig = go.Figure()

    # Base function (structural break)
    base_func = np.where(t < T / 2, 0, 1) + 0.5 * np.sin(2 * np.pi * t / T)
    fig.add_trace(go.Scatter(x=t, y=base_func, name="True Function",
                             line=dict(color='black', width=3)))

    # Fourier approximation
    fourier_approx = np.zeros_like(t)
    for k in range(1, freq_slider + 1):
        cos_comp = np.cos(2 * np.pi * k * t / T)
        sin_comp = np.sin(2 * np.pi * k * t / T)
        fourier_approx += 0.5 * cos_comp + 0.3 * sin_comp

    fig.add_trace(go.Scatter(x=t, y=fourier_approx, name=f"Fourier Approximation (n={freq_slider})",
                             line=dict(color='red', width=2, dash='dash')))

    fig.update_layout(
        title="Fourier Series Approximation of Structural Breaks",
        xaxis_title="Time",
        yaxis_title="Value",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

elif section == "Traditional vs Fourier Methods":
    st.markdown('<div class="section-header">⚖️ Traditional vs Fourier Approaches</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="theory-box">
        <h3>🏛️ Traditional Dummy Variable Approach</h3>
        <p>Traditional tests use dummy variables to account for structural breaks:</p>
        </div>
        """, unsafe_allow_html=True)

        st.latex(r'''
        \Delta y_t = \alpha + \beta y_{t-1} + \sum_{i=1}^{m} \theta_i D_{TB_i,t} + \varepsilon_t
        ''')

        st.markdown("Where:")
        st.latex(r'''
        D_{TB_i,t} = \begin{cases} 
        1 & \text{if } t = TB_i + 1 \\
        0 & \text{otherwise}
        \end{cases}
        ''')

        st.markdown("**Limitations:**")
        st.markdown("""
        - Requires knowledge of break dates
        - Limited to sharp breaks
        - Low power when break dates are unknown
        - Multiple testing issues
        """)

    with col2:
        st.markdown("""
        <div class="theory-box">
        <h3>🌊 Fourier Approach</h3>
        <p>Fourier tests use trigonometric functions:</p>
        </div>
        """, unsafe_allow_html=True)

        st.latex(r'''
        \Delta y_t = \alpha + \beta y_{t-1} + \sum_{k=1}^{n} \left[ \gamma_k \sin\left(\frac{2\pi kt}{T}\right) + \delta_k \cos\left(\frac{2\pi kt}{T}\right) \right] + \varepsilon_t
        ''')

        st.markdown("**Advantages:**")
        st.markdown("""
        - No prior knowledge of break dates needed
        - Captures both smooth and sharp breaks
        - Higher statistical power
        - Flexible break patterns
        - Single test framework
        """)

    st.markdown("""
    <div class="comparison-box">
    <h3>📊 Comparative Framework</h3>
    </div>
    """, unsafe_allow_html=True)

    # Comparison table
    comparison_data = {
        "Aspect": ["Break Date Knowledge", "Break Type", "Number of Breaks", "Statistical Power",
                   "Computational Complexity", "False Rejection Rate"],
        "Traditional Dummy": ["Required", "Sharp only", "Limited", "Low when unknown dates",
                              "High (multiple tests)", "High"],
        "Fourier Method": ["Not Required", "Smooth & Sharp", "Multiple", "High",
                           "Low (single test)", "Well-controlled"]
    }

    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)

elif section == "Fourier Unit Root Tests":
    st.markdown('<div class="section-header">🧮 Fourier Unit Root Test Specifications</div>', unsafe_allow_html=True)

    test_type = st.selectbox("Select Test Type:",
                             ["Single Frequency Fourier", "Cumulative Fourier", "Flexible Fourier"])

    if test_type == "Single Frequency Fourier":
        st.markdown("""
        <div class="equation-box">
        <h3>Single Frequency Fourier ADF Test</h3>
        </div>
        """, unsafe_allow_html=True)

        st.latex(r'''
        \Delta y_t = \alpha + \beta y_{t-1} + \gamma_1 \sin\left(\frac{2\pi t}{T}\right) + \gamma_2 \cos\left(\frac{2\pi t}{T}\right) + \sum_{j=1}^{p} \phi_j \Delta y_{t-j} + \varepsilon_t
        ''')

        st.markdown("**Test Procedure:**")
        st.markdown("""
        1. **Null Hypothesis**: H₀: β = 0 (unit root with structural break)
        2. **Alternative Hypothesis**: H₁: β < 0 (stationary with structural break)
        3. **Test Statistic**: t-ratio for β coefficient
        4. **Critical Values**: Depend on sample size and significance level
        """)

        # Single frequency simulation
        st.markdown("### 📈 Single Frequency Simulation")

        col1, col2 = st.columns([1, 2])

        with col1:
            n_obs = st.slider("Sample Size", 50, 500, 200)
            break_magnitude = st.slider("Break Magnitude", 0.0, 2.0, 1.0, 0.1)
            noise_level = st.slider("Noise Level", 0.1, 1.0, 0.5, 0.1)

        with col2:
            # Generate data
            np.random.seed(42)
            t = np.arange(1, n_obs + 1)

            # Structural break component (single frequency)
            break_component = break_magnitude * np.sin(2 * np.pi * t / n_obs)

            # Generate I(1) process with break
            innovations = np.random.normal(0, noise_level, n_obs)
            y = np.cumsum(innovations) + break_component

            fig = make_subplots(rows=2, cols=1,
                                subplot_titles=["Time Series with Fourier Break", "Break Component"])

            fig.add_trace(go.Scatter(x=t, y=y, name="Series with Break",
                                     line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=t, y=break_component, name="Fourier Component",
                                     line=dict(color='red')), row=2, col=1)

            fig.update_layout(height=500, title="Single Frequency Fourier Break")
            st.plotly_chart(fig, use_container_width=True)

    elif test_type == "Cumulative Fourier":
        st.markdown("""
        <div class="equation-box">
        <h3>Cumulative Fourier Test</h3>
        </div>
        """, unsafe_allow_html=True)

        st.latex(r'''
        \Delta y_t = \alpha + \beta y_{t-1} + \sum_{k=1}^{n} \left[ \gamma_k \sin\left(\frac{2\pi kt}{T}\right) + \delta_k \cos\left(\frac{2\pi kt}{T}\right) \right] + \sum_{j=1}^{p} \phi_j \Delta y_{t-j} + \varepsilon_t
        ''')

        st.markdown("**Key Features:**")
        st.markdown("""
        - Multiple frequencies capture complex break patterns
        - Frequency selection via information criteria
        - Higher flexibility in break modeling
        - Better approximation of unknown break functions
        """)

        # Cumulative frequency simulation
        st.markdown("### 📊 Cumulative Frequency Simulation")

        col1, col2 = st.columns([1, 2])

        with col1:
            n_obs = st.slider("Sample Size", 50, 500, 200, key="cum_size")
            n_freq = st.slider("Number of Frequencies", 1, 5, 2)
            break_type = st.selectbox("Break Pattern", ["Multiple Sharp", "Smooth Transition", "Complex"])

        with col2:
            np.random.seed(42)
            t = np.arange(1, n_obs + 1)

            # Cumulative Fourier break component
            break_component = np.zeros(n_obs)
            for k in range(1, n_freq + 1):
                if break_type == "Multiple Sharp":
                    break_component += 0.5 * np.cos(2 * np.pi * k * t / n_obs)
                elif break_type == "Smooth Transition":
                    break_component += (1 / k) * np.sin(2 * np.pi * k * t / n_obs)
                else:  # Complex
                    break_component += np.random.normal(0, 0.3) * np.sin(2 * np.pi * k * t / n_obs)

            innovations = np.random.normal(0, 0.5, n_obs)
            y = np.cumsum(innovations) + break_component

            fig = make_subplots(rows=3, cols=1,
                                subplot_titles=["Series with Cumulative Fourier Breaks",
                                                "Cumulative Break Component", "Individual Frequencies"])

            fig.add_trace(go.Scatter(x=t, y=y, name="Series", line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=t, y=break_component, name="Cumulative Break",
                                     line=dict(color='red')), row=2, col=1)

            colors = ['green', 'orange', 'purple', 'brown', 'pink']
            for k in range(1, min(n_freq + 1, 6)):
                freq_comp = np.sin(2 * np.pi * k * t / n_obs)
                fig.add_trace(go.Scatter(x=t, y=freq_comp, name=f"Freq {k}",
                                         line=dict(color=colors[k - 1])), row=3, col=1)

            fig.update_layout(height=700, title="Cumulative Fourier Analysis")
            st.plotly_chart(fig, use_container_width=True)

    else:  # Flexible Fourier
        st.markdown("""
        <div class="equation-box">
        <h3>Flexible Fourier Form (Enders & Lee, 2012)</h3>
        </div>
        """, unsafe_allow_html=True)

        st.latex(r'''
        y_t = \alpha + \beta t + \sum_{k=1}^{n} \left[ \gamma_k \sin\left(\frac{2\pi kt}{T}\right) + \delta_k \cos\left(\frac{2\pi kt}{T}\right) \right] + u_t
        ''')

        st.latex(r'''
        \Delta u_t = \rho u_{t-1} + \sum_{j=1}^{p} \phi_j \Delta u_{t-j} + \varepsilon_t
        ''')

        st.markdown("**Flexible Features:**")
        st.markdown("""
        - Separates deterministic and stochastic components
        - Optimal frequency selection
        - Robust to various break patterns
        - Improved finite sample properties
        """)

elif section == "Interactive Simulations":
    st.markdown('<div class="section-header">🎮 Interactive Simulations</div>', unsafe_allow_html=True)

    simulation_type = st.selectbox("Simulation Type:",
                                   ["Break vs No Break", "Power Analysis", "Size Distortion"])

    if simulation_type == "Break vs No Break":
        st.markdown("### 🔄 Unit Root vs Stationary with Breaks")

        col1, col2, col3 = st.columns(3)

        with col1:
            process_type = st.selectbox("Process Type:", ["Unit Root", "Stationary with Break"])
            n_obs = st.slider("Sample Size", 100, 1000, 300)

        with col2:
            if process_type == "Stationary with Break":
                ar_coef = st.slider("AR Coefficient", -0.95, -0.05, -0.5, 0.05)
                break_magnitude = st.slider("Break Size", 0.5, 3.0, 1.5, 0.1)
            else:
                ar_coef = 0.0
                break_magnitude = 1.0

        with col3:
            n_freq = st.slider("Fourier Frequencies", 1, 5, 2)
            noise_std = st.slider("Innovation Std", 0.1, 1.0, 0.5, 0.1)

        # Generate simulation
        np.random.seed(42)
        t = np.arange(1, n_obs + 1)
        innovations = np.random.normal(0, noise_std, n_obs)

        if process_type == "Unit Root":
            # Pure random walk
            y = np.cumsum(innovations)
            title_suffix = "Unit Root Process"
        else:
            # Stationary with structural break
            fourier_break = break_magnitude * np.sin(2 * np.pi * t / n_obs)
            if n_freq > 1:
                fourier_break += 0.5 * break_magnitude * np.cos(4 * np.pi * t / n_obs)

            # AR process with break
            y = np.zeros(n_obs)
            y[0] = innovations[0] + fourier_break[0]
            for i in range(1, n_obs):
                y[i] = ar_coef * y[i - 1] + fourier_break[i] + innovations[i]
            title_suffix = f"Stationary Process (φ={ar_coef:.2f})"

        # Plotting
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=[f"Generated Series - {title_suffix}",
                                            "First Differences", "ACF of Levels", "ACF of Differences"])

        # Original series
        fig.add_trace(go.Scatter(x=t, y=y, name="Series", line=dict(color='blue')), row=1, col=1)

        # First differences
        dy = np.diff(y)
        fig.add_trace(go.Scatter(x=t[1:], y=dy, name="Δy", line=dict(color='red')), row=1, col=2)

        # ACF calculations (simplified)
        max_lag = min(20, n_obs // 4)
        lags = np.arange(max_lag)

        # ACF of levels
        acf_levels = [np.corrcoef(y[:-i] if i > 0 else y, y[i:])[0, 1] if i < n_obs - 1 else 0 for i in range(max_lag)]
        fig.add_trace(go.Bar(x=lags, y=acf_levels, name="ACF Levels", marker_color='lightblue'), row=2, col=1)

        # ACF of differences
        acf_diff = [np.corrcoef(dy[:-i] if i > 0 else dy, dy[i:])[0, 1] if i < len(dy) - 1 else 0 for i in
                    range(max_lag)]
        fig.add_trace(go.Bar(x=lags, y=acf_diff, name="ACF Differences", marker_color='lightcoral'), row=2, col=2)

        fig.update_layout(height=600, title="Time Series Analysis Dashboard")
        st.plotly_chart(fig, use_container_width=True)

        # Theoretical test statistics
        st.markdown("### 📊 Theoretical Test Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Traditional ADF Test**")
            if process_type == "Unit Root":
                adf_result = "Fail to reject H₀ (Unit Root)"
                adf_color = "🔴"
            else:
                adf_result = "May fail to reject H₀ (Type II Error)"
                adf_color = "🟡"
            st.markdown(f"{adf_color} {adf_result}")

        with col2:
            st.markdown("**Fourier ADF Test**")
            if process_type == "Unit Root":
                fadf_result = "Fail to reject H₀ (Unit Root)"
                fadf_color = "🔴"
            else:
                fadf_result = "Reject H₀ (Stationary with Break)"
                fadf_color = "🟢"
            st.markdown(f"{fadf_color} {fadf_result}")

    elif simulation_type == "Power Analysis":
        st.markdown("### ⚡ Statistical Power Comparison")

        col1, col2 = st.columns(2)

        with col1:
            n_simulations = st.slider("Number of Simulations", 100, 1000, 500)
            sample_sizes = st.multiselect("Sample Sizes", [50, 100, 200, 500, 1000], [100, 200, 500])

        with col2:
            break_magnitudes = st.multiselect("Break Magnitudes", [0.5, 1.0, 1.5, 2.0, 2.5], [1.0, 2.0])
            significance_level = st.selectbox("Significance Level", [0.01, 0.05, 0.10], 1)

        if st.button("Run Power Analysis"):
            # Simulate power analysis
            results = []

            for sample_size in sample_sizes:
                for break_mag in break_magnitudes:
                    # Simulate traditional test power (approximation)
                    traditional_power = max(0.1, min(0.9, 0.3 + 0.2 * break_mag + 0.001 * sample_size))

                    # Simulate Fourier test power (higher)
                    fourier_power = max(0.2, min(0.95, 0.5 + 0.3 * break_mag + 0.0015 * sample_size))

                    results.append({
                        'Sample Size': sample_size,
                        'Break Magnitude': break_mag,
                        'Traditional ADF': traditional_power,
                        'Fourier ADF': fourier_power
                    })

            df_results = pd.DataFrame(results)

            # Create power curves
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=["Power vs Sample Size", "Power vs Break Magnitude"])

            for break_mag in break_magnitudes:
                subset = df_results[df_results['Break Magnitude'] == break_mag]

                fig.add_trace(go.Scatter(x=subset['Sample Size'],
                                         y=subset['Traditional ADF'],
                                         name=f'Traditional (Break={break_mag})',
                                         line=dict(dash='dash')), row=1, col=1)

                fig.add_trace(go.Scatter(x=subset['Sample Size'],
                                         y=subset['Fourier ADF'],
                                         name=f'Fourier (Break={break_mag})',
                                         line=dict(dash='solid')), row=1, col=1)

            for sample_size in sample_sizes:
                subset = df_results[df_results['Sample Size'] == sample_size]

                fig.add_trace(go.Scatter(x=subset['Break Magnitude'],
                                         y=subset['Traditional ADF'],
                                         name=f'Traditional (n={sample_size})',
                                         line=dict(dash='dash')), row=1, col=2)

                fig.add_trace(go.Scatter(x=subset['Break Magnitude'],
                                         y=subset['Fourier ADF'],
                                         name=f'Fourier (n={sample_size})',
                                         line=dict(dash='solid')), row=1, col=2)

            fig.update_xaxes(title_text="Sample Size", row=1, col=1)
            fig.update_xaxes(title_text="Break Magnitude", row=1, col=2)
            fig.update_yaxes(title_text="Statistical Power", row=1, col=1)
            fig.update_yaxes(title_text="Statistical Power", row=1, col=2)

            fig.update_layout(height=500, title="Power Analysis Results")
            st.plotly_chart(fig, use_container_width=True)

elif section == "Smooth vs Sharp Breaks":
    st.markdown('<div class="section-header">🌊 Smooth vs Sharp Structural Breaks</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="theory-box">
    <h3>Understanding Break Types</h3>
    <p>Fourier-based tests excel at detecting both smooth and sharp breaks, while traditional dummy variable approaches are limited to sharp breaks only.</p>
    </div>
    """, unsafe_allow_html=True)

    break_comparison = st.selectbox("Break Type Comparison:",
                                    ["Sharp Break", "Smooth Break", "Multiple Breaks", "Complex Pattern"])

    col1, col2 = st.columns([1, 2])

    with col1:
        n_obs = st.slider("Sample Size", 100, 500, 200, key="break_size")
        break_timing = st.slider("Break Timing (%)", 20, 80, 50)
        intensity = st.slider("Break Intensity", 0.5, 3.0, 1.5, 0.1)

    with col2:
        np.random.seed(42)
        t = np.arange(1, n_obs + 1)
        break_point = int(n_obs * break_timing / 100)

        if break_comparison == "Sharp Break":
            # Traditional dummy variable break
            break_component = np.where(t > break_point, intensity, 0)

            # Fourier approximation
            fourier_approx = intensity * (0.5 + (2 / np.pi) * np.arctan(np.sin(2 * np.pi * t / n_obs) /
                                                                        (1 - np.cos(2 * np.pi * t / n_obs))))

        elif break_comparison == "Smooth Break":
            # Smooth transition
            transition_speed = 20
            break_component = intensity / (1 + np.exp(-transition_speed * (t - break_point) / n_obs))

            # Fourier representation
            fourier_approx = np.zeros_like(t)
            for k in range(1, 4):
                fourier_approx += (intensity / k) * np.sin(2 * np.pi * k * t / n_obs)

        elif break_comparison == "Multiple Breaks":
            # Multiple sharp breaks
            break_points = [n_obs // 3, 2 * n_obs // 3]
            break_component = np.zeros_like(t)
            for bp in break_points:
                break_component += intensity * np.where(t > bp, 0.5, 0)

            # Fourier approximation
            fourier_approx = np.zeros_like(t)
            for k in range(1, 3):
                fourier_approx += intensity * 0.3 * np.cos(2 * np.pi * k * t / n_obs)

        else:  # Complex Pattern
            # Complex break pattern
            break_component = (intensity * np.sin(4 * np.pi * t / n_obs) *
                               np.exp(-(t - n_obs / 2) ** 2 / (2 * (n_obs / 6) ** 2)))

            # Fourier approximation
            fourier_approx = np.zeros_like(t)
            for k in range(1, 5):
                fourier_approx += (intensity / k) * np.sin(2 * np.pi * k * t / n_obs) * np.cos(np.pi * k / 2)

        # Generate full time series
        innovations = np.random.normal(0, 0.3, n_obs)
        y_true = np.cumsum(innovations) + break_component
        y_fourier = np.cumsum(innovations) + fourier_approx

        # Plotting
        fig = make_subplots(rows=3, cols=1,
                            subplot_titles=[f"Time Series with {break_comparison}",
                                            "Break Components Comparison",
                                            "Fourier Decomposition"])

        # Time series
        fig.add_trace(go.Scatter(x=t, y=y_true, name="True Series",
                                 line=dict(color='blue', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=y_fourier, name="Fourier Approximation",
                                 line=dict(color='red', width=2, dash='dash')), row=1, col=1)

        # Break components
        fig.add_trace(go.Scatter(x=t, y=break_component, name="True Break",
                                 line=dict(color='green', width=3)), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=fourier_approx, name="Fourier Break",
                                 line=dict(color='orange', width=2, dash='dot')), row=2, col=1)

        # Individual Fourier components
        colors = ['purple', 'brown', 'pink', 'gray', 'olive']
        for k in range(1, min(4, 6)):
            fourier_k = (intensity / k) * np.sin(2 * np.pi * k * t / n_obs)
            fig.add_trace(go.Scatter(x=t, y=fourier_k, name=f"Fourier {k}",
                                     line=dict(color=colors[k - 1], width=1)), row=3, col=1)

        fig.update_layout(height=700, title=f"Analysis of {break_comparison}")
        st.plotly_chart(fig, use_container_width=True)

    # Mathematical representation
    st.markdown("### 📐 Mathematical Representation")

    if break_comparison == "Sharp Break":
        st.latex(r'''
        \text{Sharp Break: } f(t) = \begin{cases} 
        0 & \text{if } t \leq TB \\
        \mu & \text{if } t > TB
        \end{cases}
        ''')

        st.latex(r'''
        \text{Fourier Approximation: } f(t) \approx \frac{\mu}{2} + \frac{2\mu}{\pi} \sum_{k=1}^{\infty} \frac{\sin(2\pi k TB/T)}{k} \cos\left(\frac{2\pi kt}{T}\right)
        ''')

    elif break_comparison == "Smooth Break":
        st.latex(r'''
        \text{Smooth Break: } f(t) = \frac{\mu}{1 + e^{-\lambda(t - TB)}}
        ''')

        st.latex(r'''
        \text{Fourier Series: } f(t) = a_0 + \sum_{k=1}^{\infty} \left[ a_k \cos\left(\frac{2\pi kt}{T}\right) + b_k \sin\left(\frac{2\pi kt}{T}\right) \right]
        ''')

elif section == "Comparative Analysis":
    st.markdown('<div class="section-header">📈 Comprehensive Comparative Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="comparison-box">
    <h3>Test Performance Metrics</h3>
    </div>
    """, unsafe_allow_html=True)

    # Performance comparison table
    performance_data = {
        "Test Characteristic": [
            "Break Date Knowledge Required",
            "Maximum Detectable Breaks",
            "Break Type Flexibility",
            "Computational Complexity",
            "Statistical Power (Known Breaks)",
            "Statistical Power (Unknown Breaks)",
            "Size Distortion",
            "Robustness to Break Timing",
            "Asymptotic Properties",
            "Finite Sample Performance"
        ],
        "Traditional ADF": [
            "No", "No", "No", "Low", "low", "Low", "Low", "Poor", "Well-established", "Good"
        ],
        "Dummy Variable Tests": [
            "Yes", "Limited", "Sharp only", "High", "High", "Medium", "Medium", "Poor", "Well-established", "Medium"
        ],
        "Single Fourier": [
            "No", "∞", "Smooth & Sharp", "Low", "Medium", "High", "Low", "Excellent", "Established", "Good"
        ],
        "Cumulative Fourier": [
            "No", "∞", "Very Flexible", "Medium", "High", "Very High", "Very Low", "Excellent", "Established",
            "Very Good"
        ],
        "Flexible Fourier": [
            "No", "∞", "Very Flexible", "Medium", "Very High", "Very High", "Very Low", "Excellent", "Recent",
            "Excellent"
        ]
    }

    df_performance = pd.DataFrame(performance_data)
    st.dataframe(df_performance, use_container_width=True)

    # Critical value comparison
    st.markdown("### 📊 Critical Values Comparison")

    significance_levels = [0.01, 0.05, 0.10]
    sample_sizes = [100, 250, 500]

    # Simulated critical values (these would come from Monte Carlo simulations)
    critical_values = {
        "Traditional ADF": {
            100: [-3.43, -2.86, -2.57],
            250: [-3.43, -2.87, -2.57],
            500: [-3.44, -2.87, -2.57]
        },
        "Single Fourier": {
            100: [-4.47, -3.84, -3.52],
            250: [-4.32, -3.72, -3.42],
            500: [-4.27, -3.69, -3.39]
        },
        "Cumulative Fourier": {
            100: [-4.85, -4.19, -3.86],
            250: [-4.71, -4.09, -3.78],
            500: [-4.65, -4.05, -3.75]
        }
    }

    # Create critical values table
    cv_data = []
    for test in critical_values.keys():
        for size in sample_sizes:
            row = {"Test": test, "Sample Size": size}
            for i, alpha in enumerate(significance_levels):
                row[f"{int(alpha * 100)}% level"] = critical_values[test][size][i]
            cv_data.append(row)

    df_cv = pd.DataFrame(cv_data)
    st.dataframe(df_cv, use_container_width=True)

    # Visualization of critical values
    fig = go.Figure()

    for test in critical_values.keys():
        cv_5pct = [critical_values[test][size][1] for size in sample_sizes]
        fig.add_trace(go.Scatter(x=sample_sizes, y=cv_5pct, name=f"{test} (5%)",
                                 mode='lines+markers', line=dict(width=3)))

    fig.update_layout(
        title="Critical Values at 5% Significance Level",
        xaxis_title="Sample Size",
        yaxis_title="Critical Value",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Final recommendations
    st.markdown("""
    <div class="theory-box">
    <h3>🎯 Practical Recommendations</h3>
    <ol>
    <li><strong>Use Fourier-based tests</strong> when break dates are unknown or when you suspect smooth transitions</li>
    <li><strong>Start with single frequency</strong> for simple break patterns, then move to cumulative for complex patterns</li>
    <li><strong>Consider sample size</strong> - Fourier tests perform better in small samples</li>
    <li><strong>Validate results</strong> by comparing different specifications</li>
    <li><strong>Use information criteria</strong> for optimal frequency selection in cumulative tests</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p>📚 This application  Created by Dr Merwan Roudane provides theoretical insights into Fourier-based unit root tests.<br>
    For empirical applications, specialized econometric software or custom implementations would be required.</p>
</div>
""", unsafe_allow_html=True)