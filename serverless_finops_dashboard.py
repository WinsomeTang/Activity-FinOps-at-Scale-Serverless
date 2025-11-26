import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Page configuration
st.set_page_config(
    page_title="RetailNova FinOps Dashboard",
    page_icon="",
    layout="wide"
)

# Title
st.title("RetailNova - FinOps at Scale for Serverless Applications")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    # Read CSV and handle quoted lines
    with open("Serverless_Data.csv", 'r') as f:
        content = f.read()
    lines = content.strip().split('\n')
    fixed_lines = [line.strip('"') for line in lines]
    fixed_content = '\n'.join(fixed_lines)

    from io import StringIO
    df = pd.read_csv(StringIO(fixed_content))
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
environments = st.sidebar.multiselect(
    "Select Environment",
    options=df['Environment'].unique(),
    default=df['Environment'].unique()
)

# Filter data
filtered_df = df[df['Environment'].isin(environments)]

# Overview metrics
st.header("Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Functions", len(filtered_df))
with col2:
    st.metric("Total Monthly Cost", f"${filtered_df['CostUSD'].sum():,.2f}")
with col3:
    st.metric("Total Invocations", f"{filtered_df['InvocationsPerMonth'].sum():,.0f}")
with col4:
    st.metric("Total GB-Seconds", f"{filtered_df['GBSeconds'].sum():,.2f}")

st.markdown("---")

# =============================================================================
# EXERCISE 1: Identify Top Cost Contributors
# =============================================================================
st.header("Exercise 1: Top Cost Contributors")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Functions Contributing 80% of Total Spend")

    # Sort by cost descending
    sorted_df = filtered_df.sort_values('CostUSD', ascending=False).copy()
    sorted_df['CumulativeCost'] = sorted_df['CostUSD'].cumsum()
    sorted_df['CumulativePercent'] = (sorted_df['CumulativeCost'] / sorted_df['CostUSD'].sum()) * 100

    # Find functions contributing to 80%
    top_contributors = sorted_df[sorted_df['CumulativePercent'] <= 80].copy()
    # Include next function to cross 80%
    if len(top_contributors) < len(sorted_df):
        next_idx = len(top_contributors)
        top_contributors = pd.concat([top_contributors, sorted_df.iloc[[next_idx]]])

    st.write(f"**{len(top_contributors)} functions** contribute 80% of total spend (${top_contributors['CostUSD'].sum():,.2f})")

    # Display table
    display_cols = ['FunctionName', 'Environment', 'CostUSD', 'CumulativePercent']
    st.dataframe(
        top_contributors[display_cols].rename(columns={
            'CostUSD': 'Cost ($)',
            'CumulativePercent': 'Cumulative %'
        }).style.format({'Cost ($)': '${:.2f}', 'Cumulative %': '{:.1f}%'}),
        use_container_width=True
    )

with col2:
    st.subheader("Cost vs Invocation Frequency")

    fig = px.scatter(
        filtered_df,
        x='InvocationsPerMonth',
        y='CostUSD',
        color='Environment',
        size='MemoryMB',
        hover_data=['FunctionName', 'AvgDurationMs'],
        title="Cost vs Invocations (bubble size = Memory)",
        labels={'InvocationsPerMonth': 'Invocations/Month', 'CostUSD': 'Cost ($)'}
    )
    fig.update_layout(xaxis_type='log')
    st.plotly_chart(fig, use_container_width=True)

# Pareto chart
st.subheader("Pareto Analysis - Cost Distribution")
pareto_df = filtered_df.sort_values('CostUSD', ascending=False).head(20)
pareto_df['CumulativeCost'] = pareto_df['CostUSD'].cumsum()
pareto_df['CumulativePercent'] = (pareto_df['CumulativeCost'] / filtered_df['CostUSD'].sum()) * 100

fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
fig_pareto.add_trace(
    go.Bar(x=pareto_df['FunctionName'], y=pareto_df['CostUSD'], name="Cost ($)"),
    secondary_y=False
)
fig_pareto.add_trace(
    go.Scatter(x=pareto_df['FunctionName'], y=pareto_df['CumulativePercent'],
               name="Cumulative %", mode='lines+markers', line=dict(color='red')),
    secondary_y=True
)
fig_pareto.update_layout(title="Top 20 Functions by Cost (Pareto Chart)", xaxis_tickangle=-45)
fig_pareto.update_yaxes(title_text="Cost ($)", secondary_y=False)
fig_pareto.update_yaxes(title_text="Cumulative %", secondary_y=True)
st.plotly_chart(fig_pareto, use_container_width=True)

st.markdown("---")

# =============================================================================
# EXERCISE 2: Memory Right-Sizing
# =============================================================================
st.header("Exercise 2: Memory Right-Sizing Analysis")

st.subheader("Functions with Low Duration but High Memory (Over-provisioned)")

# Define thresholds
duration_threshold = st.slider("Max Duration (ms) for 'low duration'", 100, 1000, 500)
memory_threshold = st.slider("Min Memory (MB) for 'high memory'", 256, 2048, 1024)

# Find over-provisioned functions
over_provisioned = filtered_df[
    (filtered_df['AvgDurationMs'] < duration_threshold) &
    (filtered_df['MemoryMB'] >= memory_threshold)
].copy()

col1, col2 = st.columns(2)

with col1:
    st.write(f"**{len(over_provisioned)} functions** identified as potentially over-provisioned")

    if len(over_provisioned) > 0:
        # Calculate potential savings (assuming 50% memory reduction)
        over_provisioned['RecommendedMemoryMB'] = over_provisioned['MemoryMB'] / 2
        over_provisioned['EstimatedNewCost'] = over_provisioned['CostUSD'] * 0.6  # Rough estimate
        over_provisioned['PotentialSavings'] = over_provisioned['CostUSD'] - over_provisioned['EstimatedNewCost']

        display_df = over_provisioned[['FunctionName', 'Environment', 'AvgDurationMs',
                                        'MemoryMB', 'RecommendedMemoryMB', 'CostUSD', 'PotentialSavings']]
        st.dataframe(
            display_df.style.format({
                'CostUSD': '${:.2f}',
                'PotentialSavings': '${:.2f}'
            }),
            use_container_width=True
        )

        total_savings = over_provisioned['PotentialSavings'].sum()
        st.success(f"**Total Potential Monthly Savings: ${total_savings:,.2f}**")

with col2:
    # Scatter plot: Duration vs Memory with cost as color
    fig_mem = px.scatter(
        filtered_df,
        x='AvgDurationMs',
        y='MemoryMB',
        color='CostUSD',
        size='InvocationsPerMonth',
        hover_data=['FunctionName'],
        title="Duration vs Memory (color = cost, size = invocations)",
        color_continuous_scale='Reds'
    )
    # Add threshold lines
    fig_mem.add_hline(y=memory_threshold, line_dash="dash", line_color="green",
                      annotation_text=f"Memory threshold: {memory_threshold}MB")
    fig_mem.add_vline(x=duration_threshold, line_dash="dash", line_color="blue",
                      annotation_text=f"Duration threshold: {duration_threshold}ms")
    st.plotly_chart(fig_mem, use_container_width=True)

st.markdown("---")

# =============================================================================
# EXERCISE 3: Provisioned Concurrency Optimization
# =============================================================================
st.header("Exercise 3: Provisioned Concurrency Optimization")

# Functions with provisioned concurrency
pc_functions = filtered_df[filtered_df['ProvisionedConcurrency'] > 0].copy()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Functions with Provisioned Concurrency")

    if len(pc_functions) > 0:
        # Estimate PC cost (approximately $0.000004463 per GB-second for PC)
        pc_functions['EstimatedPCCost'] = pc_functions['ProvisionedConcurrency'] * pc_functions['MemoryMB'] / 1024 * 720 * 0.000004463 * 3600
        pc_functions['ColdStartBenefit'] = pc_functions['ColdStartRate'].apply(
            lambda x: 'High' if x >= 0.05 else ('Medium' if x >= 0.02 else 'Low')
        )

        # Recommendation
        pc_functions['Recommendation'] = pc_functions.apply(
            lambda row: 'Keep PC' if row['ColdStartRate'] >= 0.03 or row['InvocationsPerMonth'] > 1000000
            else ('Reduce PC' if row['ColdStartRate'] >= 0.01 else 'Remove PC'),
            axis=1
        )

        display_pc = pc_functions[['FunctionName', 'Environment', 'ColdStartRate',
                                   'ProvisionedConcurrency', 'InvocationsPerMonth', 'Recommendation']]
        st.dataframe(display_pc, use_container_width=True)

        # Summary
        recommendations = pc_functions['Recommendation'].value_counts()
        st.write("**Recommendations Summary:**")
        for rec, count in recommendations.items():
            st.write(f"- {rec}: {count} functions")

with col2:
    st.subheader("Cold Start Rate vs Cost")

    fig_pc = px.scatter(
        pc_functions,
        x='ColdStartRate',
        y='CostUSD',
        size='ProvisionedConcurrency',
        color='Recommendation',
        hover_data=['FunctionName', 'InvocationsPerMonth'],
        title="Cold Start Rate vs Cost (size = PC units)"
    )
    fig_pc.update_xaxes(tickformat='.1%')
    st.plotly_chart(fig_pc, use_container_width=True)

st.markdown("---")

# =============================================================================
# EXERCISE 4: Detect Unused or Low-Value Workloads
# =============================================================================
st.header("Exercise 4: Unused or Low-Value Workloads")

total_invocations = filtered_df['InvocationsPerMonth'].sum()
invocation_threshold = total_invocations * 0.01  # 1% of total

# Find low-value functions
filtered_df_copy = filtered_df.copy()
filtered_df_copy['InvocationPercent'] = (filtered_df_copy['InvocationsPerMonth'] / total_invocations) * 100

# Low invocation but relatively high cost
median_cost = filtered_df_copy['CostUSD'].median()
low_value = filtered_df_copy[
    (filtered_df_copy['InvocationPercent'] < 1) &
    (filtered_df_copy['CostUSD'] > median_cost)
].copy()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Low-Value Functions (<1% invocations, above median cost)")
    st.write(f"**{len(low_value)} functions** identified as low-value workloads")
    st.write(f"Total cost of low-value functions: **${low_value['CostUSD'].sum():,.2f}**")

    if len(low_value) > 0:
        display_lv = low_value[['FunctionName', 'Environment', 'InvocationsPerMonth',
                                'InvocationPercent', 'CostUSD', 'AvgDurationMs']].sort_values('CostUSD', ascending=False)
        st.dataframe(
            display_lv.style.format({
                'InvocationPercent': '{:.4f}%',
                'CostUSD': '${:.2f}'
            }),
            use_container_width=True
        )

with col2:
    # Visualization
    fig_lv = px.scatter(
        filtered_df_copy,
        x='InvocationPercent',
        y='CostUSD',
        color='Environment',
        hover_data=['FunctionName'],
        title="Invocation % vs Cost (identify low-value in bottom-right quadrant)"
    )
    fig_lv.add_vline(x=1, line_dash="dash", line_color="red", annotation_text="1% threshold")
    fig_lv.add_hline(y=median_cost, line_dash="dash", line_color="green",
                     annotation_text=f"Median cost: ${median_cost:.2f}")
    st.plotly_chart(fig_lv, use_container_width=True)

st.markdown("---")

# =============================================================================
# EXERCISE 5: Cost Forecasting Model
# =============================================================================
st.header("Exercise 5: Cost Forecasting Model")

st.markdown("""
**Model Formula:** Cost = Invocations x Duration x Memory x PricingCoefficients + DataTransfer
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Training")

    # Prepare features
    X = filtered_df[['InvocationsPerMonth', 'AvgDurationMs', 'MemoryMB', 'GBSeconds', 'DataTransferGB']].copy()
    y = filtered_df['CostUSD']

    # Add computed features
    X['ComputeMetric'] = X['InvocationsPerMonth'] * X['AvgDurationMs'] * X['MemoryMB'] / 1e9

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.write(f"**Model Performance:**")
    st.write(f"- R-squared: {r2:.4f}")
    st.write(f"- Mean Absolute Error: ${mae:.2f}")

    st.write("**Feature Coefficients:**")
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    st.dataframe(coef_df)

with col2:
    st.subheader("Predicted vs Actual Cost")

    # Full predictions
    y_full_pred = model.predict(X)

    fig_pred = px.scatter(
        x=y,
        y=y_full_pred,
        labels={'x': 'Actual Cost ($)', 'y': 'Predicted Cost ($)'},
        title="Predicted vs Actual Cost"
    )
    fig_pred.add_trace(go.Scatter(x=[0, y.max()], y=[0, y.max()],
                                   mode='lines', name='Perfect Prediction',
                                   line=dict(dash='dash', color='red')))
    st.plotly_chart(fig_pred, use_container_width=True)

# Cost prediction calculator
st.subheader("Cost Predictor")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    pred_invocations = st.number_input("Invocations/Month", value=100000, step=10000)
with col2:
    pred_duration = st.number_input("Avg Duration (ms)", value=500, step=50)
with col3:
    pred_memory = st.number_input("Memory (MB)", value=512, step=128)
with col4:
    pred_gb_seconds = st.number_input("GB-Seconds", value=10.0, step=1.0)
with col5:
    pred_data_transfer = st.number_input("Data Transfer (GB)", value=20.0, step=5.0)

pred_compute_metric = pred_invocations * pred_duration * pred_memory / 1e9
pred_input = pd.DataFrame([[pred_invocations, pred_duration, pred_memory,
                            pred_gb_seconds, pred_data_transfer, pred_compute_metric]],
                          columns=X.columns)
predicted_cost = model.predict(pred_input)[0]
st.success(f"**Predicted Monthly Cost: ${predicted_cost:,.2f}**")

st.markdown("---")

# =============================================================================
# EXERCISE 6: Containerization Candidates
# =============================================================================
st.header("Exercise 6: Workloads Suitable for Containerization")

st.markdown("""
**Criteria for containerization candidates:**
- Long-running: > 3 seconds (3000ms)
- High memory: > 2GB (2048MB)
- Low invocation frequency (relative)
""")

# Define thresholds
duration_threshold_container = 3000  # 3 seconds
memory_threshold_container = 2048  # 2GB
invocation_median = filtered_df['InvocationsPerMonth'].median()

# Find candidates
container_candidates = filtered_df[
    (filtered_df['AvgDurationMs'] > duration_threshold_container) |
    (filtered_df['MemoryMB'] > memory_threshold_container)
].copy()

# Score candidates
container_candidates['LongRunning'] = container_candidates['AvgDurationMs'] > duration_threshold_container
container_candidates['HighMemory'] = container_candidates['MemoryMB'] > memory_threshold_container
container_candidates['LowInvocations'] = container_candidates['InvocationsPerMonth'] < invocation_median
container_candidates['ContainerScore'] = (
    container_candidates['LongRunning'].astype(int) +
    container_candidates['HighMemory'].astype(int) +
    container_candidates['LowInvocations'].astype(int)
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Containerization Candidates")

    top_candidates = container_candidates.sort_values('ContainerScore', ascending=False)

    display_container = top_candidates[['FunctionName', 'Environment', 'AvgDurationMs',
                                         'MemoryMB', 'InvocationsPerMonth', 'CostUSD', 'ContainerScore']]
    st.dataframe(
        display_container.style.format({'CostUSD': '${:.2f}'}).background_gradient(
            subset=['ContainerScore'], cmap='Reds'
        ),
        use_container_width=True
    )

    st.write(f"**{len(container_candidates)} functions** identified as containerization candidates")
    st.write(f"Potential cost impact: **${container_candidates['CostUSD'].sum():,.2f}** (current serverless cost)")

with col2:
    # Visualization
    fig_container = px.scatter(
        filtered_df,
        x='AvgDurationMs',
        y='MemoryMB',
        color='InvocationsPerMonth',
        size='CostUSD',
        hover_data=['FunctionName', 'Environment'],
        title="Duration vs Memory (color = invocations, size = cost)",
        color_continuous_scale='Blues_r'
    )
    fig_container.add_hline(y=memory_threshold_container, line_dash="dash", line_color="red",
                            annotation_text="2GB threshold")
    fig_container.add_vline(x=duration_threshold_container, line_dash="dash", line_color="red",
                            annotation_text="3s threshold")
    # Highlight quadrant
    fig_container.add_shape(
        type="rect",
        x0=duration_threshold_container, y0=memory_threshold_container,
        x1=filtered_df['AvgDurationMs'].max() * 1.1, y1=filtered_df['MemoryMB'].max() * 1.1,
        fillcolor="red", opacity=0.1,
        line=dict(width=0)
    )
    st.plotly_chart(fig_container, use_container_width=True)

st.markdown("---")

# =============================================================================
# Summary & Recommendations
# =============================================================================
st.header("Summary & Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Cost Distribution by Environment")
    env_costs = filtered_df.groupby('Environment')['CostUSD'].sum().reset_index()
    fig_env = px.pie(env_costs, values='CostUSD', names='Environment',
                     title="Cost by Environment")
    st.plotly_chart(fig_env, use_container_width=True)

with col2:
    st.subheader("Key Findings")

    total_cost = filtered_df['CostUSD'].sum()
    prod_cost = filtered_df[filtered_df['Environment'] == 'production']['CostUSD'].sum()

    st.markdown(f"""
    1. **Total Monthly Serverless Cost:** ${total_cost:,.2f}
    2. **Production vs Non-Production:** {prod_cost/total_cost*100:.1f}% in production
    3. **Top Cost Contributors:** {len(top_contributors)} functions make up 80% of spend
    4. **Over-provisioned Functions:** {len(over_provisioned)} functions could reduce memory
    5. **Low-Value Workloads:** {len(low_value)} functions with <1% invocations but high cost
    6. **Containerization Candidates:** {len(container_candidates)} functions may be better suited for containers
    """)

    # Estimated savings
    estimated_savings = over_provisioned['PotentialSavings'].sum() if len(over_provisioned) > 0 else 0
    st.success(f"**Estimated Potential Savings: ${estimated_savings:,.2f}/month** (from memory right-sizing alone)")

# Footer
st.markdown("---")
st.caption("RetailNova FinOps Dashboard | INFO49971 Cloud Economics | Fall 2025")
