"""
streamlit_visualization.py
Contains the Streamlit visualization script to explore the parameter search results.

Usage:
    From the project root directory:
    streamlit run backend/scripts/streamlit_visualization.py --server.headless true

    Troubleshooting:
    export PYTHONPATH=/path/to/your/project:$PYTHONPATH
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from backend.app.core.config import ROOT_DIR

# Set Seaborn style
sns.set(style="whitegrid")
sns.set_palette("colorblind")

######################################################
#       1) DATA LOADING AND CACHING                 #
######################################################

@st.cache_data
def load_data(filepath: Path, file_format: str = "csv", nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load data from a CSV or Parquet file.
    """
    if not filepath.exists():
        st.error(f"File does not exist: {filepath}")
        return pd.DataFrame()

    try:
        if file_format == "csv":
            df = pd.read_csv(filepath, nrows=nrows)
        elif file_format == "parquet":
            df = pd.read_parquet(filepath)
        else:
            st.error(f"Unknown file format: {file_format}")
            return pd.DataFrame()

        st.write(f"Loaded data: shape={df.shape} from {filepath.name}")
        return df
    except Exception as e:
        st.error(f"Error loading file '{filepath}': {e}")
        return pd.DataFrame()
    
def get_processed_data(selected_k: int) -> pd.DataFrame:
    """
    Load and process data for the given selected_k.
    This function is cached so that re-running the app with changed UI parameters
    does not re-read and reprocess the large CSV unless selected_k changes.
    """
    DATA_DIR = Path(ROOT_DIR)/ "data" / "experiments_param_search"
    csv_path = DATA_DIR / f"parameter_search_comparison_topk_{selected_k}.csv"
    df = load_data(filepath=csv_path, file_format="csv")
    if df.empty:
        return df
    return process_data(df)  # process_data is also cached

######################################################
#         2) DATA PROCESSING FUNCTIONS              #
######################################################

@st.cache_data
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the loaded DataFrame by parsing JSON, exploding detailed results,
    and extracting relevant information.
    """
    df_copy = df.copy()

    # Parse 'detailed_results' JSON strings
    def parse_json(x):
        try:
            return json.loads(x)
        except (TypeError, json.JSONDecodeError):
            return None

    if df_copy['detailed_results'].apply(lambda x: isinstance(x, str)).any():
        df_copy['detailed_results'] = df_copy['detailed_results'].apply(parse_json)
        st.write("Parsed 'detailed_results' from JSON strings to Python objects.")
    else:
        st.write("'detailed_results' is already in a parsed format.")

    # Reset index to create a unique identifier for each parameter combination
    df_copy = df_copy.reset_index().rename(columns={'index': 'combo_id'})

    # Explode 'detailed_results' so that each test case is a separate row
    df_exploded = df_copy.explode('detailed_results').reset_index(drop=True)

    # Drop rows where 'detailed_results' is None (failed to parse)
    df_exploded = df_exploded[df_exploded['detailed_results'].notnull()]

    st.write(f"Exploded DataFrame shape: {df_exploded.shape}")

    # Assign a test_case_number within each combo_id
    df_exploded['test_case_number'] = df_exploded.groupby('combo_id').cumcount() + 1

    return df_exploded

def extract_test_case(df_exploded: pd.DataFrame, test_case_index: int) -> pd.DataFrame:
    """
    Extract and process data for a specific test case index.
    """
    df_specific_test_case = df_exploded[df_exploded['test_case_number'] == test_case_index].reset_index(drop=True)

    if df_specific_test_case.empty:
        st.warning(f"No data found for test_case_index={test_case_index}.")
        return df_specific_test_case

    # Extract 'test_case' and 'top_diseases' from the exploded JSON
    df_specific_test_case['test_case'] = df_specific_test_case['detailed_results'].apply(lambda x: x.get('test_case', {}))
    df_specific_test_case['top_diseases'] = df_specific_test_case['detailed_results'].apply(lambda x: x.get('top_diseases', []))

    st.write(f"DataFrame for Test Case {test_case_index} shape: {df_specific_test_case.shape}")

    return df_specific_test_case

@st.cache_data
def get_top_diseases(df_specific_test_case: pd.DataFrame, top_j: int) -> pd.DataFrame:
    """
    Explode 'top_diseases' and retain only the top J diseases by frequency.
    """
    # Explode 'top_diseases' so each disease is a separate row
    df_top_diseases = df_specific_test_case.explode('top_diseases').reset_index(drop=True).copy()

    # Extract 'disease' and 'score' from each top disease
    df_top_diseases['disease'] = df_top_diseases['top_diseases'].apply(lambda x: x.get('disease'))
    df_top_diseases['score'] = df_top_diseases['top_diseases'].apply(lambda x: x.get('score'))

    # Drop the original 'top_diseases' and 'detailed_results' columns
    df_top_diseases = df_top_diseases.drop(columns=['top_diseases', 'detailed_results'])

    # Determine the top J diseases by frequency
    top_diseases = df_top_diseases['disease'].value_counts().head(top_j).index.tolist()

    # Filter the DataFrame to include only top J diseases
    df_top_diseases = df_top_diseases[df_top_diseases['disease'].isin(top_diseases)]

    st.write(f"Retained top {top_j} diseases. New shape={df_top_diseases.shape}")

    # Display the frequency of top diseases
    st.write(f"### Top {top_j} Diseases (Frequency)")
    st.bar_chart(df_top_diseases['disease'].value_counts().head(top_j))

    return df_top_diseases

def filter_parameters(df: pd.DataFrame,
                      base_weight: Optional[List[float]] = None,
                      local_weight: Optional[List[float]] = None,
                      global_weight: Optional[List[float]] = None,
                      param_coocc: Optional[List[float]] = None,
                      param_pmi: Optional[List[float]] = None,
                      param_npmi: Optional[List[float]] = None,
                      param_llr: Optional[List[float]] = None,
                      modes: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Filters the DataFrame based on selected parameter values.
    """
    filtered_df = df.copy()

    if base_weight is not None:
        filtered_df = filtered_df[filtered_df['base_weight'].isin(base_weight)]
        st.write(f"Filtered by base_weight = {base_weight}: {filtered_df.shape[0]} rows remaining.")

    if local_weight is not None:
        filtered_df = filtered_df[filtered_df['local_weight'].isin(local_weight)]
        st.write(f"Filtered by local_weight = {local_weight}: {filtered_df.shape[0]} rows remaining.")

    if global_weight is not None:
        filtered_df = filtered_df[filtered_df['global_weight'].isin(global_weight)]
        st.write(f"Filtered by global_weight = {global_weight}: {filtered_df.shape[0]} rows remaining.")

    if param_coocc is not None:
        filtered_df = filtered_df[filtered_df['param_coocc'].isin(param_coocc)]
        st.write(f"Filtered by param_coocc = {param_coocc}: {filtered_df.shape[0]} rows remaining.")

    if param_pmi is not None:
        filtered_df = filtered_df[filtered_df['param_pmi'].isin(param_pmi)]
        st.write(f"Filtered by param_pmi = {param_pmi}: {filtered_df.shape[0]} rows remaining.")

    if param_npmi is not None:
        filtered_df = filtered_df[filtered_df['param_npmi'].isin(param_npmi)]
        st.write(f"Filtered by param_npmi = {param_npmi}: {filtered_df.shape[0]} rows remaining.")

    if param_llr is not None:
        filtered_df = filtered_df[filtered_df['param_llr'].isin(param_llr)]
        st.write(f"Filtered by param_llr = {param_llr}: {filtered_df.shape[0]} rows remaining.")

    if modes is not None:
        filtered_df = filtered_df[filtered_df['mode'].isin(modes)]
        st.write(f"Filtered by modes = {modes}: {filtered_df.shape[0]} rows remaining.")

    st.write(f"**Final filtered DataFrame shape:** {filtered_df.shape}")
    return filtered_df

######################################################
#         3) PLOTTING FUNCTIONS                      #
######################################################

def plot_scatter_by_mode(long_df: pd.DataFrame):
    """
    Creates a facet grid with one scatter plot per 'mode'.
    Each subplot shows 'score' vs. 'combo_id', colored by disease.
    """
    if 'mode' not in long_df.columns:
        st.error("The 'mode' column is missing from the data.")
        return

    g = sns.FacetGrid(long_df, col="mode", col_wrap=2, height=4, sharex=False, sharey=True)
    g.map_dataframe(
        sns.scatterplot,
        x="combo_id",
        y="score",
        hue="disease",
        alpha=0.5,   # Consistent transparency
        s=35         # Updated marker size
    )
    g.add_legend(title='Top-j Diseases', bbox_to_anchor=(1, 0.9), loc='upper right')
    legend = g._legend
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(10)
        legend.get_title().set_fontsize(12)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Scatter Plots: Combo ID vs. Score, Faceted by Mode")
    st.pyplot(g.fig)


def plot_scatter_faceted_by_param_conditions(long_df: pd.DataFrame):
    """
    Creates scatter plots faceted by specific parameter conditions.
    """
    conditions = [
        (long_df['param_pmi'] == 1) & (long_df['param_npmi'] == 0) & (long_df['param_llr'] == 0),
        (long_df['param_npmi'] == 1) & (long_df['param_pmi'] == 0) & (long_df['param_llr'] == 0),
        (long_df['param_llr'] == 1) & (long_df['param_pmi'] == 0) & (long_df['param_npmi'] == 0),
        (long_df['param_pmi'] == 0) & (long_df['param_npmi'] == 0) & (long_df['param_llr'] == 0)
    ]

    labels = [
        'param_pmi=1',
        'param_npmi=1',
        'param_llr=1',
        'All Params=0'
    ]

    long_df = long_df.copy()
    long_df['param_condition'] = 'Other'
    for condition, label in zip(conditions, labels):
        long_df.loc[condition, 'param_condition'] = label

    df_filtered = long_df[long_df['param_condition'] != 'Other']

    if df_filtered.empty:
        st.warning("No data matches the specified parameter conditions.")
        return

    g = sns.FacetGrid(
        df_filtered,
        col="param_condition",
        col_wrap=2,
        height=4,
        sharex=False,
        sharey=True
    )
    g.map_dataframe(
        sns.scatterplot,
        x="combo_id",
        y="score",
        hue="disease",
        alpha=0.5,
        s=35
    )
    g.add_legend(title='Top-j Diseases', bbox_to_anchor=(1, 0.9), loc='upper right')
    legend = g._legend
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(10)
        legend.get_title().set_fontsize(12)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Scatter Plots: Combo ID vs. Score, Faceted by Parameter Conditions")
    st.pyplot(g.fig)


def plot_scatter_faceted_by_weight_conditions(long_df: pd.DataFrame, tolerance: float = 1e-4):
    """
    Creates scatter plots faceted by specific weight conditions with tolerance for floating-point comparisons.
    """
    conditions = [
        np.isclose(long_df['base_weight'], 1.0, atol=tolerance) &
        np.isclose(long_df['local_weight'], 0.0, atol=tolerance) &
        np.isclose(long_df['global_weight'], 0.0, atol=tolerance),

        np.isclose(long_df['base_weight'], 0.0, atol=tolerance) &
        np.isclose(long_df['local_weight'], 1.0, atol=tolerance) &
        np.isclose(long_df['global_weight'], 0.0, atol=tolerance),

        np.isclose(long_df['base_weight'], 0.0, atol=tolerance) &
        np.isclose(long_df['local_weight'], 0.0, atol=tolerance) &
        np.isclose(long_df['global_weight'], 1.0, atol=tolerance),

        np.isclose(long_df['base_weight'], 0.5, atol=tolerance) &
        np.isclose(long_df['local_weight'], 0.25, atol=tolerance) &
        np.isclose(long_df['global_weight'], 0.25, atol=tolerance)
    ]

    labels = [
        'Base Weight=1.0, Others=0.0',
        'Local Weight=1.0, Others=0.0',
        'Global Weight=1.0, Others=0.0',
        'Base=0.5, Local=0.25, Global=0.25'
    ]
    
    long_df['weight_condition'] = np.select(conditions, labels, default='Other')

    df_weight_conditions = long_df.copy()
    for condition, label in zip(conditions, labels):
        df_weight_conditions.loc[condition, 'weight_condition'] = label

    df_filtered = df_weight_conditions  # Using all rows as before

    if df_filtered.empty:
        st.warning("No data matches the specified weight conditions.")
        return

    # Define the desired facet order:
    col_order = [
        'Base Weight=1.0, Others=0.0',         # Upper left: base
        'Global Weight=1.0, Others=0.0',         # Upper right: global
        'Local Weight=1.0, Others=0.0',          # Lower left: local
        'Base=0.5, Local=0.25, Global=0.25'       # Lower right: mixed
    ]

    g = sns.FacetGrid(
        df_filtered,
        col="weight_condition",
        col_order=col_order,
        col_wrap=2,
        height=4,
        sharex=False,
        sharey=True
    )
    g.map_dataframe(
        sns.scatterplot,
        x="combo_id",
        y="score",
        hue="disease",
        alpha=0.5,
        s=35
    )
    g.add_legend(title='Top-j Diseases', bbox_to_anchor=(1, 0.9), loc='upper right')
    legend = g._legend
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(10)
        legend.get_title().set_fontsize(12)

    # Remove the "weight_condition = " prefix from each subplot title.
    for ax in g.axes.flatten():
        title = ax.get_title()  # e.g., "weight_condition = Base Weight=1.0, Others=0.0"
        if "weight_condition = " in title:
            new_title = title.replace("weight_condition = ", "")
            ax.set_title(new_title)

    plt.subplots_adjust(top=0.91)
    g.fig.suptitle("Scatter Plots: Combo ID vs. Score, Faceted by Weight Conditions")
    st.pyplot(g.fig)


def plot_heatmap(long_df: pd.DataFrame):
    """
    Creates a heatmap of Diseases vs. Parameter Combo IDs.
    """
    pivot_df = long_df.pivot_table(
        index="disease", 
        columns="combo_id", 
        values="score", 
        aggfunc="mean"
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_df, 
        cmap="YlGnBu"
    )
    plt.title("Heatmap: Diseases vs Parameter Combo IDs (mean score)")
    plt.xlabel("Combo ID")
    plt.ylabel("Disease")
    plt.tight_layout()
    st.pyplot(plt)


def plot_violin_by_disease(long_df: pd.DataFrame):
    """
    Shows the distribution of scores for each disease.
    """
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=long_df, 
        x="disease", 
        y="score", 
        inner="box",
        palette="Pastel1"
    )
    plt.title("Violin Plot: Score Distributions per Disease")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)


def plot_disease_rank_line(long_df: pd.DataFrame):
    """
    Plots disease ranks across parameter combos as a line plot.
    """
    df_ranked = long_df.copy()
    df_ranked["rank"] = df_ranked.groupby("combo_id")["score"].rank(
        method="dense", ascending=False
    )
    # Average rank per combo_id per disease
    df_line = df_ranked.groupby(["combo_id", "disease"])["rank"].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_line, x="combo_id", y="rank", hue="disease", marker="o")
    plt.title("Disease Rank Line Plot: Combo ID vs Average Rank")
    plt.xlabel("Combo ID")
    plt.ylabel("Average Rank")
    plt.gca().invert_yaxis()  # So rank=1 is at the top
    plt.legend(prop={'size': 10}, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    st.pyplot(plt)


def plot_scatter_small_points(long_df: pd.DataFrame):
    """
    Plots disease scores by combo_id with scatter points (using the same style), all modes together.
    """
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=long_df.sort_values("combo_id"), 
        x="combo_id", 
        y="score", 
        hue="disease",
        alpha=0.5,
        s=35
    )
    plt.title("Scatter Plot: Combo ID vs Score (all modes)")
    plt.xlabel("Combo ID")
    plt.ylabel("Score")
    plt.legend(prop={'size': 10}, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    st.pyplot(plt)


######################################################
#         4) MAIN STREAMLIT APPLICATION             #
######################################################

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Disease Score Visualization Dashboard")
    st.write("Welcome to the Disease Score Visualization Dashboard.")

    # -- Sidebar Inputs --
    st.sidebar.header("1) Select Top K")

    # Define the data directory
    DATA_DIR = Path(ROOT_DIR)/ "data" / "experiments_param_search"

    if not DATA_DIR.exists():
        st.error(f"Data directory '{DATA_DIR}' does not exist.")
        return

    # List available CSV files
    available_csv_files = sorted(DATA_DIR.glob("parameter_search_comparison_topk_*.csv"))
    available_ks = [int(file.stem.split('_')[-1]) for file in available_csv_files]

    if not available_ks:
        st.error("No parameter search CSV files found in the data directory.")
        return

    # Sidebar widget for Top K selection
    selected_k = st.sidebar.selectbox(
        "Select Top K",
        options=available_ks,
        index=available_ks.index(5) if 5 in available_ks else 0
    )

    st.write(f"**Selected Top K:** {selected_k}")

    # Load the selected CSV file
    csv_path = DATA_DIR / f"parameter_search_comparison_topk_{selected_k}.csv"
    df = load_data(filepath=csv_path, file_format="csv")

    if df.empty:
        st.warning("Loaded DataFrame is empty.")
        return

    # Process the data
    df_exploded = get_processed_data(selected_k)
    if df_exploded.empty:
        st.warning("Loaded DataFrame is empty.")
        return

    # -- Sidebar: Select Test Case --
    st.sidebar.header("2) Select Test Case")

    max_test_case = df_exploded['test_case_number'].max()
    test_case_index = st.sidebar.number_input(
        "Test Case Index (1-based)",
        min_value=1,
        max_value=int(max_test_case),
        value=5,
        step=1
    )

    df_specific_test_case = extract_test_case(df_exploded, test_case_index)

    if df_specific_test_case.empty:
        st.warning(f"No data found for test_case_index={test_case_index}.")
        return

    # -- Sidebar: Select Top J Diseases --
    st.sidebar.header("3) Select Top J Diseases")
    top_j = st.sidebar.slider("Number of Top Diseases to Retain (J)", min_value=5, max_value=50, value=10, step=1)
    df_top_diseases = get_top_diseases(df_specific_test_case, top_j)

    if df_top_diseases.empty:
        st.warning("No diseases available after selecting top J.")
        return

    # -- Sidebar: Filter Parameters --
    st.sidebar.header("4) Filter Parameters")

    # Get unique values for each parameter (for slider/select inputs)
    parameter_columns = ['base_weight', 'local_weight', 'global_weight', 
                         'param_coocc', 'param_pmi', 'param_npmi', 'param_llr', 'mode']
    unique_values = {col: sorted(df_top_diseases[col].unique()) for col in parameter_columns if col in df_top_diseases.columns}

    # --- Base Weight ---
    st.sidebar.subheader("Base Weight")
    base_weight_choice = st.sidebar.radio("Base Weight Filter", ["All", "Select Value"], key="base_weight_filter")
    if base_weight_choice == "Select Value":
        selected_base_weight = st.sidebar.select_slider(
            "Select Base Weight", 
            options=unique_values.get("base_weight", []),
            value=unique_values.get("base_weight", [0])[0],
            key="base_weight_slider"
        )
    else:
        selected_base_weight = None

    # --- Local Weight ---
    st.sidebar.subheader("Local Weight")
    if selected_base_weight is None:
        lw_options = unique_values.get("local_weight", [])
    else:
        lw_options = [lw for lw in unique_values.get("local_weight", []) if (selected_base_weight + lw <= 1.0)]
    local_weight_choice = st.sidebar.radio("Local Weight Filter", ["All", "Select Value"], key="local_weight_filter")
    if local_weight_choice == "Select Value" and lw_options:
        selected_local_weight = st.sidebar.select_slider(
            "Select Local Weight", 
            options=lw_options,
            value=lw_options[0],
            key="local_weight_slider"
        )
    else:
        selected_local_weight = None

    # --- Global Weight ---
    st.sidebar.subheader("Global Weight")
    if selected_base_weight is not None and selected_local_weight is not None:
        gw_calculated = 1.0 - selected_base_weight - selected_local_weight
        if gw_calculated in unique_values.get("global_weight", []):
            gw_options = [gw_calculated]
        else:
            gw_options = unique_values.get("global_weight", [])
    else:
        gw_options = unique_values.get("global_weight", [])
    global_weight_choice = st.sidebar.radio("Global Weight Filter", ["All", "Select Value"], key="global_weight_filter")
    if global_weight_choice == "Select Value" and gw_options:
        selected_global_weight = st.sidebar.select_slider(
            "Select Global Weight", 
            options=gw_options,
            value=gw_options[0],
            key="global_weight_slider"
        )
    else:
        selected_global_weight = None

    # --- Param Coocc ---
    st.sidebar.subheader("Param Coocc")
    param_coocc_choice = st.sidebar.radio("Param Coocc Filter", ["All", "Select Value"], key="param_coocc_filter")
    if param_coocc_choice == "Select Value":
        selected_param_coocc = st.sidebar.select_slider(
            "Select Param Coocc", 
            options=unique_values.get("param_coocc", []),
            value=unique_values.get("param_coocc", [0])[0],
            key="param_coocc_slider"
        )
    else:
        selected_param_coocc = None

    # --- Param PMI ---
    st.sidebar.subheader("Param PMI")
    param_pmi_choice = st.sidebar.radio("Param PMI Filter", ["All", "Select Value"], key="param_pmi_filter")
    if param_pmi_choice == "Select Value":
        selected_param_pmi = st.sidebar.select_slider(
            "Select Param PMI", 
            options=unique_values.get("param_pmi", []),
            value=unique_values.get("param_pmi", [0])[0],
            key="param_pmi_slider"
        )
    else:
        selected_param_pmi = None

    # --- Param NPMI ---
    st.sidebar.subheader("Param NPMI")
    param_npmi_choice = st.sidebar.radio("Param NPMI Filter", ["All", "Select Value"], key="param_npmi_filter")
    if param_npmi_choice == "Select Value":
        selected_param_npmi = st.sidebar.select_slider(
            "Select Param NPMI", 
            options=unique_values.get("param_npmi", []),
            value=unique_values.get("param_npmi", [0])[0],
            key="param_npmi_slider"
        )
    else:
        selected_param_npmi = None

    # --- Param LLR ---
    st.sidebar.subheader("Param LLR")
    param_llr_choice = st.sidebar.radio("Param LLR Filter", ["All", "Select Value"], key="param_llr_filter")
    if param_llr_choice == "Select Value":
        selected_param_llr = st.sidebar.select_slider(
            "Select Param LLR", 
            options=unique_values.get("param_llr", []),
            value=unique_values.get("param_llr", [0])[0],
            key="param_llr_slider"
        )
    else:
        selected_param_llr = None

    # --- Mode ---
    st.sidebar.subheader("Mode")
    mode_choice = st.sidebar.radio("Mode Filter", ["All", "Select Value"], key="mode_filter")
    if mode_choice == "Select Value":
        selected_mode = st.sidebar.selectbox(
            "Select Mode", 
            options=unique_values.get("mode", []),
            key="mode_select"
        )
    else:
        selected_mode = None

    # (No need for "View Available Parameters" expander now.)

    # Display selected parameter values (if desired)
    with st.sidebar.expander("View Selected Parameters"):
        st.write(f"**Base Weight:** {selected_base_weight if selected_base_weight is not None else 'All'}")
        st.write(f"**Local Weight:** {selected_local_weight if selected_local_weight is not None else 'All'}")
        st.write(f"**Global Weight:** {selected_global_weight if selected_global_weight is not None else 'All'}")
        st.write(f"**Param Coocc:** {selected_param_coocc if selected_param_coocc is not None else 'All'}")
        st.write(f"**Param PMI:** {selected_param_pmi if selected_param_pmi is not None else 'All'}")
        st.write(f"**Param NPMI:** {selected_param_npmi if selected_param_npmi is not None else 'All'}")
        st.write(f"**Param LLR:** {selected_param_llr if selected_param_llr is not None else 'All'}")
        st.write(f"**Mode:** {selected_mode if selected_mode is not None else 'All'}")
        st.write(f"**Top Diseases to Retain:** {top_j}")
        st.write(f"**Test Case Index:** {test_case_index}")

    # Apply filters (passing lists if a specific value is chosen)
    filtered_long_df = filter_parameters(
        df=df_top_diseases,
        base_weight=[selected_base_weight] if selected_base_weight is not None else None,
        local_weight=[selected_local_weight] if selected_local_weight is not None else None,
        global_weight=[selected_global_weight] if selected_global_weight is not None else None,
        param_coocc=[selected_param_coocc] if selected_param_coocc is not None else None,
        param_pmi=[selected_param_pmi] if selected_param_pmi is not None else None,
        param_npmi=[selected_param_npmi] if selected_param_npmi is not None else None,
        param_llr=[selected_param_llr] if selected_param_llr is not None else None,
        modes=[selected_mode] if selected_mode is not None else None,
    )

    if filtered_long_df.empty:
        st.warning("No data available after applying the selected filters.")
        return

    # -- Display Test Case Symptoms (always open) --
    sample_test_case = df_specific_test_case['test_case'].iloc[0]
    symptoms = sample_test_case.get("symptoms", "No symptoms available")
    st.subheader("Test Case Symptoms")
    if isinstance(symptoms, list):
        st.markdown("### Symptoms")
        for idx, symptom in enumerate(symptoms, start=1):
            if isinstance(symptom, list):
                st.write(f"{idx}: {', '.join(map(str, symptom))}")
            else:
                st.write(f"{idx}: {symptom}")
    else:
        st.write("Symptoms not available.")

    # -- Display Available Visualizations --
    st.header("5) Available Visualizations")
    # Reordered plot options:
    plot_options = [
        "Violin Plot: Score Distributions per Disease",
        "Scatter Plot: Combo ID vs. Score, Faceted by Weight Conditions",
        "Scatter Plot Faceted by Parameter Conditions",
        "Scatter Plot: Combo ID vs Score (all modes)",
        "Disease Rank Line Plot",
        "Heatmap: Diseases vs Parameter Combo IDs"
    ]

    # All plots are selected by default
    selected_plots = st.multiselect("Select Plots to Display", plot_options, default=plot_options[:4])

    if "Violin Plot: Score Distributions per Disease" in selected_plots:
        st.subheader("Violin Plot: Score Distributions per Disease")
        plot_violin_by_disease(filtered_long_df)

    if "Scatter Plot: Combo ID vs. Score, Faceted by Weight Conditions" in selected_plots:
        st.subheader("Scatter Plot: Combo ID vs. Score, Faceted by Weight Conditions")
        plot_scatter_faceted_by_weight_conditions(filtered_long_df)
    
    if "Scatter Plot: Combo ID vs Score (all modes)" in selected_plots:
        st.subheader("Scatter Plot: Combo ID vs Score (all modes)")
        plot_scatter_small_points(filtered_long_df)

    if "Scatter Plot Faceted by Parameter Conditions" in selected_plots:
        st.subheader("Scatter Plot: Combo ID vs. Score, Faceted by Parameter Conditions")
        plot_scatter_faceted_by_param_conditions(filtered_long_df)

    if "Heatmap: Diseases vs Parameter Combo IDs" in selected_plots:
        st.subheader("Heatmap: Diseases vs Parameter Combo IDs (mean score)")
        plot_heatmap(filtered_long_df)

    if "Disease Rank Line Plot" in selected_plots:
        st.subheader("Disease Rank Line Plot: Combo ID vs Average Rank")
        plot_disease_rank_line(filtered_long_df)

    if not selected_plots:
        st.info("Please select at least one plot to display.")

    st.success("All selected visualizations have been generated successfully.")

if __name__ == "__main__":
    main()
