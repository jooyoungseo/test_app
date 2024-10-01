import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import maidr

# Set random seed
np.random.seed(1000)

# Define color palettes
color_palettes = {
    "Default": "#007bc2",
    "Red": "#FF0000",
    "Green": "#00FF00",
    "Blue": "#0000FF",
    "Purple": "#800080",
    "Orange": "#FFA500"
}

# Helper function to set theme
def set_theme(fig, ax):
    theme = st.session_state.get('theme', 'Light')
    if theme == "Dark":
        plt.style.use('dark_background')
        fig.patch.set_facecolor('#2E2E2E')
        ax.set_facecolor('#2E2E2E')
    else:
        plt.style.use('default')
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

# Sidebar for user input
st.sidebar.title("Settings")
theme = st.sidebar.selectbox("Theme:", ["Light", "Dark"])
st.session_state['theme'] = theme

# Sliders to adjust figure size (now in the sidebar)
fig_width = st.sidebar.slider("Figure width", min_value=5, max_value=15, value=10)
fig_height = st.sidebar.slider("Figure height", min_value=3, max_value=10, value=6)

# Main content
st.title("Learning Data Visualization with MAIDR")

# Tabs for different plots
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Practice", "Histogram", "Box Plot", "Scatter Plot", "Bar Plot", "Line Plot", "Heatmap"
])

# Function to render plots using Maidr
def render_maidr_plot(plot):
    # Render the plot using maidr and display it in Streamlit
    components.html(
        maidr.render(
            plot
        ).get_html_string(),
        scrolling=True,
        height=fig_height * 100,
        width=fig_width * 100,
    )

# Practice tab: Allows users to upload a CSV and generate plots
# Practice tab: Allows users to upload a CSV and generate plots
with tab1:
    st.header("Practice with your own data")
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:", df.head())
        
        # Select the plot type
        plot_type = st.selectbox("Select plot type:", [
            "Histogram", "Box Plot", "Scatter Plot", "Bar Plot", "Line Plot", "Heatmap"
        ])
        
        # Color palette selection
        plot_color = st.selectbox("Select plot color:", list(color_palettes.keys()))
        
        # Select columns from uploaded data for plots
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

        if plot_type == "Histogram":
            var = st.selectbox("Select numeric variable for histogram:", numeric_columns)
            if var:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                set_theme(fig, ax)
                sns.histplot(data=df, x=var, kde=True, color=color_palettes[plot_color], ax=ax)
                ax.set_title(f"{var}")
                ax.set_xlabel(var)
                render_maidr_plot(ax)

        elif plot_type == "Box Plot":
            var_x = st.selectbox("Select numerical variable for X-axis:", numeric_columns)
            var_y = st.selectbox("Select categorical variable for Y-axis (optional):", [""] + categorical_columns)
            if var_x:
                fig, ax = plt.subplots(figsize=(10, 6))
                set_theme(fig, ax)
                if var_y:
                    sns.boxplot(x=var_y, y=var_x, data=df, palette=[color_palettes[plot_color]], ax=ax)
                    ax.set_title(f"{var_x} grouped by {var_y}")
                    ax.set_xlabel(var_y.replace("_", " ").title())
                    ax.set_ylabel(var_x.replace("_", " ").title())
                else:
                    sns.boxplot(y=df[var_x], color=color_palettes[plot_color], ax=ax)
                    ax.set_title(f"{var_x}")
                    ax.set_ylabel(var_x.replace("_", " ").title())
                render_maidr_plot(ax)

        elif plot_type == "Scatter Plot":
            x_var = st.selectbox("Select X variable:", numeric_columns)
            y_var = st.selectbox("Select Y variable:", [col for col in numeric_columns if col != x_var])
            if x_var and y_var:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                set_theme(fig, ax)
                sns.scatterplot(data=df, x=x_var, y=y_var, color=color_palettes[plot_color], ax=ax)
                ax.set_title(f"{x_var} vs {y_var}")
                render_maidr_plot(ax)

        elif plot_type == "Bar Plot":
            var = st.selectbox("Select categorical variable for bar plot:", categorical_columns)
            if var:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                set_theme(fig, ax)
                sns.countplot(x=var, data=df, color=color_palettes[plot_color], ax=ax)
                ax.set_title(f"{var}")
                render_maidr_plot(ax)

        elif plot_type == "Line Plot":
            x_var = st.selectbox("Select X variable:", numeric_columns)
            y_var = st.selectbox("Select Y variable:", [col for col in numeric_columns if col != x_var])
            if x_var and y_var:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                set_theme(fig, ax)
                sns.lineplot(data=df, x=x_var, y=y_var, color=color_palettes[plot_color], ax=ax)
                ax.set_title(f"{x_var} vs {y_var}")
                render_maidr_plot(ax)

        elif plot_type == "Heatmap":
            x_var = st.selectbox("Select X variable:", numeric_columns)
            y_var = st.selectbox("Select Y variable:", [col for col in numeric_columns if col != x_var])
            if x_var and y_var:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                set_theme(fig, ax)
                sns.heatmap(pd.crosstab(df[x_var], df[y_var]), ax=ax, cmap="YlGnBu", annot=True)
                ax.set_title(f"{x_var} vs {y_var}")
                render_maidr_plot(ax)

# Histogram tab
with tab2:
    st.header("Histogram")
    
    hist_type = st.selectbox("Select histogram distribution type:", [
        "Normal Distribution", "Positively Skewed", "Negatively Skewed",
        "Unimodal Distribution", "Bimodal Distribution", "Multimodal Distribution"
    ])
    hist_color = st.selectbox("Select histogram color:", list(color_palettes.keys()), key="hist_color")

    # Generate data based on user selection
    def hist_data():
        if hist_type == "Normal Distribution":
            return np.random.normal(size=1000)
        elif hist_type == "Positively Skewed":
            return np.random.exponential(scale=3, size=1000)
        elif hist_type == "Negatively Skewed":
            return -np.random.exponential(scale=1.5, size=1000)
        elif hist_type == "Unimodal Distribution":
            return np.random.normal(loc=0, scale=2.5, size=1000)
        elif hist_type == "Bimodal Distribution":
            return np.concatenate([np.random.normal(-2, 0.5, size=500), np.random.normal(2, 0.5, size=500)])
        elif hist_type == "Multimodal Distribution":
            return np.concatenate([np.random.normal(-2, 0.5, size=300), np.random.normal(2, 0.5, size=300), np.random.normal(5, 0.5, size=400)])

    # Plot the histogram using Matplotlib
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    set_theme(fig, ax)
    sns.histplot(hist_data(), kde=True, bins=20, color=color_palettes[hist_color], edgecolor="white", ax=ax)
    ax.set_title(f"{hist_type}")
    ax.set_xlabel(hist_type)
    ax.set_ylabel("Count")

    # Render using Maidr
    render_maidr_plot(ax)

# Box Plot tab
with tab3:
    st.header("Box Plot")

    box_type = st.selectbox("Select box plot type:", [
        "Positively Skewed with Outliers", "Negatively Skewed with Outliers",
        "Symmetric with Outliers", "Symmetric without Outliers"
    ])
    box_color = st.selectbox("Select box plot color:", list(color_palettes.keys()), key="box_color")

    def box_data():
        if box_type == "Positively Skewed with Outliers":
            return np.random.lognormal(mean=0, sigma=0.5, size=1000)
        elif box_type == "Negatively Skewed with Outliers":
            return -np.random.lognormal(mean=0, sigma=0.5, size=1000)
        elif box_type == "Symmetric with Outliers":
            return np.random.normal(loc=0, scale=1, size=1000)
        elif box_type == "Symmetric without Outliers":
            data = np.random.normal(loc=0, scale=1, size=1000)
            return data[(data > -1.5) & (data < 1.5)]

    # Plot the box plot using Matplotlib
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    set_theme(fig, ax)
    sns.boxplot(x=box_data(), ax=ax, color=color_palettes[box_color])
    ax.set_title(f"{box_type}")

    # Render using Maidr
    render_maidr_plot(ax)

# Scatter Plot tab
with tab4:
    st.header("Scatter Plot")

    scatter_type = st.selectbox("Select scatter plot type:", [
        "No Correlation", "Weak Positive Correlation", "Strong Positive Correlation",
        "Weak Negative Correlation", "Strong Negative Correlation"
    ])
    scatter_color = st.selectbox("Select scatter plot color:", list(color_palettes.keys()), key="scatter_color")

    def scatter_data():
        num_points = np.random.randint(20, 31)
        x = np.random.uniform(size=num_points)
        if scatter_type == "No Correlation":
            y = np.random.uniform(size=num_points)
        elif scatter_type == "Weak Positive Correlation":
            y = 0.3 * x + np.random.uniform(size=num_points)
        elif scatter_type == "Strong Positive Correlation":
            y = 0.9 * x + np.random.uniform(size=num_points) * 0.1
        elif scatter_type == "Weak Negative Correlation":
            y = -0.3 * x + np.random.uniform(size=num_points)
        elif scatter_type == "Strong Negative Correlation":
            y = -0.9 * x + np.random.uniform(size=num_points) * 0.1
        return x, y

    # Plot the scatter plot using Matplotlib
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    set_theme(fig, ax)
    data_x, data_y = scatter_data()
    sns.scatterplot(x=data_x, y=data_y, ax=ax, color=color_palettes[scatter_color])
    ax.set_title(f"{scatter_type}")

    # Render using Maidr
    render_maidr_plot(ax)

# Bar Plot tab
with tab5:
    st.header("Bar Plot")

    bar_color = st.selectbox("Select bar plot color:", list(color_palettes.keys()), key="bar_color")

    def bar_data():
        categories = ["Category A", "Category B", "Category C", "Category D", "Category E"]
        values = np.random.randint(10, 100, size=5)
        return categories, values

    # Plot the bar plot using Matplotlib
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    set_theme(fig, ax)
    categories, values = bar_data()
    sns.barplot(x=categories, y=values, ax=ax, color=color_palettes[bar_color])
    ax.set_title("Plot of Categories")

    # Render using Maidr
    render_maidr_plot(ax)

# Line Plot tab
with tab6:
    st.header("Line Plot")

    line_type = st.selectbox("Select line plot type:", [
        "Linear Trend", "Exponential Growth", "Sinusoidal Pattern", "Random Walk"
    ])
    line_color = st.selectbox("Select line plot color:", list(color_palettes.keys()), key="line_color")

    def line_data():
        x = np.linspace(0, 10, 20)
        if line_type == "Linear Trend":
            y = 2 * x + 1 + np.random.normal(0, 1, 20)
        elif line_type == "Exponential Growth":
            y = np.exp(0.5 * x) + np.random.normal(0, 1, 20)
        elif line_type == "Sinusoidal Pattern":
            y = 5 * np.sin(x) + np.random.normal(0, 0.5, 20)
        elif line_type == "Random Walk":
            y = np.cumsum(np.random.normal(0, 1, 20))
        return x, y

    # Plot the line plot using Matplotlib
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    set_theme(fig, ax)
    data_x, data_y = line_data()
    sns.lineplot(x=data_x, y=data_y, ax=ax, color=color_palettes[line_color])
    ax.set_title(f"{line_type}")

    # Render using Maidr
    render_maidr_plot(ax)

# Heatmap tab
with tab7:
    st.header("Heatmap")

    heatmap_type = st.selectbox("Select heatmap type:", [
        "Random", "Correlated", "Checkerboard"
    ])

    def heatmap_data():
        if heatmap_type == "Random":
            return np.random.rand(5, 5)
        elif heatmap_type == "Correlated":
            return np.random.multivariate_normal([0] * 5, np.eye(5), size=5)
        elif heatmap_type == "Checkerboard":
            return np.indices((5, 5)).sum(axis=0) % 2

    # Plot the heatmap using Matplotlib
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    set_theme(fig, ax)
    sns.heatmap(heatmap_data(), ax=ax, cmap="YlGnBu", annot=True, fmt=".2f")
    ax.set_title(f"{heatmap_type}")

    # Render using Maidr
    render_maidr_plot(ax)
