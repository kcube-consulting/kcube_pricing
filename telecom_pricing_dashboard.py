import streamlit as st
import pandas as pd
try:
    import plotly.express as px
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "plotly"])
    import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
from datetime import date
from scipy import interpolate
from fpdf import FPDF
import base64

# ========================================
# CONSTANTS AND CONFIGURATION
# ========================================
DEFAULT_CHAT_AGENT_COST = 2400  # For 50,000 sessions
DEFAULT_EMAIL_AGENT_COST = 1200  # For 20,000 emails
BASE_CHAT_RATE = DEFAULT_CHAT_AGENT_COST / 50000  # $0.048 per session
BASE_EMAIL_RATE = DEFAULT_EMAIL_AGENT_COST / 20000  # $0.06 per email

# Configure page
st.set_page_config(
    page_title="Kcube Pricing Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("Kcube Consulting Partners - Pricing Options (Fixed & PayG)")

# ========================================
# HELPER FUNCTIONS
# ========================================
def calculate_multi_year_costs(base_cost, years, growth_rate=0.0):
    """Calculate costs over multiple years with optional growth rate"""
    return [base_cost * (1 + growth_rate)**year for year in range(years)]

def extract_cost(value):
    """Robust cost extraction that handles all number formats"""
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Remove all non-numeric characters except decimal point and minus sign
        num_str = re.sub(r'[^\d.-]', '', value)
        if not num_str:
            return 0.0
        try:
            return float(num_str)
        except ValueError:
            return 0.0
    return 0.0
    
def format_currency(value, show_inr, exchange_rate):
    """Format currency value with INR conversion if needed"""
    if show_inr:
        return f"${value:,.2f} (₹{value*exchange_rate:,.2f})"
    return f"${value:,.2f}"

def format_telephony_cost(value, outbound_enabled):
    """Format telephony cost with * if outbound is enabled"""
    return f"${value:,.2f}{'*' if outbound_enabled else ''}"

def calculate_chat_cost(sessions):
    """Calculate chat agent cost based on sessions"""
    return sessions * BASE_CHAT_RATE

def calculate_email_cost(emails):
    """Calculate email agent cost based on emails"""
    return emails * BASE_EMAIL_RATE

# ========================================
# DATA PROCESSING FUNCTIONS
# ========================================
@st.cache_data
def load_excel_data(uploaded_file):
    """Load and process Excel data with both options"""
    try:
        xls = pd.ExcelFile(uploaded_file)
        
        # Verify sheets exist
        required_sheets = ['Option_1', 'Option_2']
        missing_sheets = [sheet for sheet in required_sheets if sheet not in xls.sheet_names]
        if missing_sheets:
            raise ValueError(f"Missing required sheets: {missing_sheets}")

        def process_sheet(df, option_name):
            """Process individual sheet to extract costs by agent count"""
            original_agents = [1,5,10,25,50]
            
            results = []
            for _, row in df.iterrows():
                metric = str(row.iloc[0]).strip()
                if 'total cost' in metric.lower():
                    monthly_costs = [extract_cost(row.iloc[i*2+1]) for i in range(len(original_agents))]
                    yearly_costs = [extract_cost(row.iloc[i*2+2]) for i in range(len(original_agents))]
                    
                    monthly_interp = interpolate.interp1d(
                        original_agents, monthly_costs,
                        kind='linear', fill_value='extrapolate'
                    )
                    yearly_interp = interpolate.interp1d(
                        original_agents, yearly_costs,
                        kind='linear', fill_value='extrapolate'
                    )
                    
                    all_agents = [1,5,10,15,20,25,30,40,50]
                    for agent in all_agents:
                        if agent in original_agents:
                            idx = original_agents.index(agent)
                            monthly = monthly_costs[idx]
                            yearly = yearly_costs[idx]
                        else:
                            monthly = float(monthly_interp(agent))
                            yearly = float(yearly_interp(agent))
                        
                        results.append({
                            'Agents': agent,
                            'MonthlyCost': monthly,
                            'YearlyCost': yearly,
                            'Option': option_name
                        })
                    break
            
            return pd.DataFrame(results)

        # Load Option 1 (read without headers)
        df1 = pd.read_excel(xls, sheet_name='Option_1', header=None)
        df1_processed = process_sheet(df1, 'Fixed Pricing')
        
        # Load Option 2 (read without headers)
        df2 = pd.read_excel(xls, sheet_name='Option_2', header=None)
        df2_processed = process_sheet(df2, 'Pay-As-You-Go')
        
        return pd.concat([df1_processed, df2_processed])
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return pd.DataFrame()

def clean_currency_columns(df):
    """Clean all currency columns in dataframe"""
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column contains currency values
            if df[col].astype(str).str.contains(r'[\$,]').any():
                df[col] = df[col].replace(r'[^\d.-]', '', regex=True).astype(float)
    return df

def process_pricing_data(df, chat_sessions, email_volume, outbound_toggle):
    """Process pricing data with chat and email costs"""
    # First ensure all cost columns are properly converted
    for col in ['MonthlyCost', 'YearlyCost']:
        if col in df.columns:
            df[col] = df[col].apply(extract_cost)
    
    # Rest of your processing logic...
    chat_cost = calculate_chat_cost(chat_sessions)
    email_cost = calculate_email_cost(email_volume)
    
    df['MonthlyCost'] = df['MonthlyCost'] * (1.1 if outbound_toggle else 1.0)
    df['YearlyCost'] = df['YearlyCost'] * (1.1 if outbound_toggle else 1.0)
    
    df['MonthlyCost'] = df['MonthlyCost'] + chat_cost + email_cost
    df['YearlyCost'] = df['YearlyCost'] + (chat_cost*12) + (email_cost*12)
    
    return df

def create_detailed_cost_table(df, agent_count, chat_sessions, email_volume, outbound_toggle):
    """Create detailed cost breakdown table with proper column names"""
    # Filter for selected agent count
    fixed = df[(df['Agents'] == agent_count) & (df['Option'] == 'Fixed Pricing')]
    payg = df[(df['Agents'] == agent_count) & (df['Option'] == 'Pay-As-You-Go')]
    
    # Create detailed table with proper column structure
    metrics = [
        "Agent's",
        "Total Hours",
        "Total Minutes",
        "In/Out Bound Telephony Cost *",
        "AI Costs",
        "Tech Infra",
        "Operational & Support Cost",
        "Cost/Call (@ 5 mins)",
        "Cost/Minute",
        "Implementation Cost",
        "Overage Rate / Hour",
        "Chat Agent",
        "Email Agent",
        "Total Cost (Excl. Implementation)"
    ]
    
    # Create data with proper column names
    data = {
        'Metric': metrics,
        'Fixed_Monthly': [
            fixed['Agents'].values[0],
            fixed['Agents'].values[0] * 120,
            fixed['Agents'].values[0] * 7200,
            format_telephony_cost(fixed['MonthlyCost'].values[0] * 0.9, outbound_toggle),
            "Included",
            "Included",
            "Included",
            f"${(fixed['MonthlyCost'].values[0] / (fixed['Agents'].values[0] * 7200 / 5)):.2f}",
            f"${(fixed['MonthlyCost'].values[0] / (fixed['Agents'].values[0] * 7200)):.2f}",
            "$15,000 (One-time)",
            "$2.50 (Min. slab)",
            f"${calculate_chat_cost(chat_sessions):,.2f} (Optional)",
            f"${calculate_email_cost(email_volume):,.2f} (Optional)",
            f"${fixed['MonthlyCost'].values[0]:,.2f}"
        ],
        'Fixed_Yearly': [
            fixed['Agents'].values[0],
            fixed['Agents'].values[0] * 1440,
            fixed['Agents'].values[0] * 86400,
            format_telephony_cost(fixed['YearlyCost'].values[0] * 0.9, outbound_toggle),
            "Included",
            "Included",
            "Included",
            f"${(fixed['YearlyCost'].values[0] / (fixed['Agents'].values[0] * 86400 / 5)):.2f}",
            f"${(fixed['YearlyCost'].values[0] / (fixed['Agents'].values[0] * 86400)):.2f}",
            "$15,000 (One-time)",
            "$2.50 (Min. slab)",
            f"${calculate_chat_cost(chat_sessions)*12:,.2f} (Optional)",
            f"${calculate_email_cost(email_volume)*12:,.2f} (Optional)",
            f"${fixed['YearlyCost'].values[0]:,.2f}"
        ],
        'PAYG_Monthly': [
            payg['Agents'].values[0],
            payg['Agents'].values[0] * 120,
            payg['Agents'].values[0] * 7200,
            format_telephony_cost(payg['MonthlyCost'].values[0] * 0.9, outbound_toggle),
            "Included",
            "Included",
            "Included",
            f"${(payg['MonthlyCost'].values[0] / (payg['Agents'].values[0] * 7200 / 5)):.2f}",
            f"${(payg['MonthlyCost'].values[0] / (payg['Agents'].values[0] * 7200)):.2f}",
            "$15,000 (One-time)",
            "$2.50 (Min. slab)",
            f"${calculate_chat_cost(chat_sessions):,.2f} (Optional)",
            f"${calculate_email_cost(email_volume):,.2f} (Optional)",
            f"${payg['MonthlyCost'].values[0]:,.2f}"
        ],
        'PAYG_Yearly': [
            payg['Agents'].values[0],
            payg['Agents'].values[0] * 1440,
            payg['Agents'].values[0] * 86400,
            format_telephony_cost(payg['YearlyCost'].values[0] * 0.9, outbound_toggle),
            "Included",
            "Included",
            "Included",
            f"${(payg['YearlyCost'].values[0] / (payg['Agents'].values[0] * 86400 / 5)):.2f}",
            f"${(payg['YearlyCost'].values[0] / (payg['Agents'].values[0] * 86400)):.2f}",
            "$15,000 (One-time)",
            "$2.50 (Min. slab)",
            f"${calculate_chat_cost(chat_sessions)*12:,.2f} (Optional)",
            f"${calculate_email_cost(email_volume)*12:,.2f} (Optional)",
            f"${payg['YearlyCost'].values[0]:,.2f}"
        ]
    }
    
    return pd.DataFrame(data)

# ========================================
# REPORT GENERATION FUNCTIONS
# ========================================
def create_detailed_pdf(data, config, usd_to_inr):
    """Generate detailed PDF report with all metrics"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Kcube Detailed Pricing Report", ln=1, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Generated on {date.today().strftime('%Y-%m-%d')}", ln=1, align='C')
    pdf.ln(10)
    
    # Configuration
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Configuration:", ln=1)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"- Agent Count: {config['agent_count']}", ln=1)
    pdf.cell(200, 10, txt=f"- Time Period: {config['time_period']}", ln=1)
    pdf.cell(200, 10, txt=f"- Outbound Included: {'Yes (+10%)*' if config['outbound'] else 'No'}", ln=1)
    pdf.cell(200, 10, txt=f"- Chat Sessions: {config['chat_sessions']:,.0f}", ln=1)
    pdf.cell(200, 10, txt=f"- Email Volume: {config['email_volume']:,.0f}", ln=1)
    pdf.cell(200, 10, txt=f"- Exchange Rate: 1 USD = {usd_to_inr} INR", ln=1)
    pdf.ln(10)
    
    # Detailed Cost Breakdown
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Detailed Cost Breakdown:", ln=1)
    pdf.set_font("Arial", size=10)
    
    # Create table header
    pdf.cell(90, 10, txt="Item Description (Metric)", border=1)
    pdf.cell(50, 10, txt="Monthly Cost", border=1)
    pdf.cell(50, 10, txt="Yearly Cost", border=1)
    pdf.ln()
    
    # Add rows to table
    for _, row in data.iterrows():
        pdf.cell(90, 10, txt=str(row['Metric']), border=1)
        pdf.cell(50, 10, txt=str(row['Fixed_Monthly']), border=1)
        pdf.cell(50, 10, txt=str(row['Fixed_Yearly']), border=1)
        pdf.ln()
    
    # Add totals
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(90, 10, txt="Total Cost (Excl. Implementation)", border=1)
    total_monthly = data[data['Metric'] == 'Total Cost (Excl. Implementation)']['Fixed_Monthly'].values[0]
    total_yearly = data[data['Metric'] == 'Total Cost (Excl. Implementation)']['Fixed_Yearly'].values[0]
    pdf.cell(50, 10, txt=str(total_monthly), border=1)
    pdf.cell(50, 10, txt=str(total_yearly), border=1)
    pdf.ln()
    
    # Recommendation
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Recommendation:", ln=1)
    pdf.set_font("Arial", size=10)
    
    fixed_cost = float(re.sub(r'[^\d.]', '', total_monthly if config['time_period'] == 'Monthly' else total_yearly))
    payg_cost = float(re.sub(r'[^\d.]', '', data[data['Metric'] == 'Total Cost (Excl. Implementation)']['PAYG_Monthly'].values[0] if config['time_period'] == 'Monthly' else data[data['Metric'] == 'Total Cost (Excl. Implementation)']['PAYG_Yearly'].values[0]))
    
    if payg_cost < fixed_cost:
        savings = fixed_cost - payg_cost
        breakeven = (15000 / savings) if savings > 0 else 0
        pdf.multi_cell(200, 10, txt=f"Choose Pay-As-You-Go and save ${savings:,.2f} {config['time_period'].lower()} (Breakeven in {breakeven:.1f} months)")
    else:
        savings = payg_cost - fixed_cost
        pdf.multi_cell(200, 10, txt=f"Choose Fixed Pricing and save ${savings:,.2f} {config['time_period'].lower()}")
    
    # Add footnote if outbound is enabled
    if config['outbound']:
        pdf.ln(5)
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(200, 5, txt="* Includes 10% outbound dialing surcharge (customer-provided dialer)", ln=1)
    
    return pdf

def get_pdf_download_link(pdf, filename):
    """Generate download link for PDF"""
    from io import BytesIO
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF Report</a>'

def display_pricing_data(df, outbound_toggle):
    """Display dataframe with proper currency formatting"""
    st.dataframe(
        df,
        column_config={
            "MonthlyCost": st.column_config.NumberColumn(
                "Monthly Cost",
                format=f"${'*' if outbound_toggle else ''}"
            ),
            "YearlyCost": st.column_config.NumberColumn(
                "Yearly Cost",
                format=f"${'*' if outbound_toggle else ''}"
            )
        },
        use_container_width=True
    )
    
def clean_numeric_data(df):
    """Convert all numeric columns from strings to actual numbers"""
    for col in df.select_dtypes(include=['object']):
        try:
            # Remove all non-numeric characters except decimal point
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True))
        except ValueError:
            continue
    return df
    
def format_currency_with_asterisk(value, show_asterisk):
    """Format currency value with optional asterisk"""
    return f"${value:,.2f}{'*' if show_asterisk else ''}" if pd.notnull(value) else value

def format_for_display(df, outbound_flag):
    """Format numeric columns for display with asterisk condition"""
    display_df = df.copy()
    formatter = lambda x: format_currency_with_asterisk(x, outbound_flag)
    
    for col in ['MonthlyCost', 'YearlyCost']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(formatter)
    return display_df
    
# ========================================
# MAIN DASHBOARD INTERFACE
# ========================================
def main():
    # File uploader
    uploaded_file = st.file_uploader("Upload Pricing Excel File", type=["xlsx"])
    if not uploaded_file:
        st.warning("Please upload your pricing Excel file")
        return

    # Load and process data
    df = load_excel_data(uploaded_file)
    df = clean_currency_columns(df)    
    
    if df.empty:
        st.error("Failed to load data. Please check the file format and structure.")
        return

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Currency settings
        usd_to_inr = st.number_input(
            "USD to INR Exchange Rate",
            min_value=1.0,
            value=87.0,
            step=0.5
        )
        show_inr = st.toggle("Show INR Pricing", value=False)
        
        # Main configuration
        agent_options = [1,5,10,15,20,25,30,40,50]
        selected_agent = st.selectbox("Select Agent Count", options=agent_options, index=2)
        outbound_toggle = st.toggle("Include Outbound (+10%)*", True)
        time_period = st.radio("Time Period", ["Monthly", "Yearly"], index=0)

        # multi-year controls
        st.subheader("Multi-Year Analysis")
        analysis_years = st.slider("Years to Analyze", 1, 5, 1)
        growth_rate = st.slider("Annual Growth Rate (%)", 0.0, 20.0, 5.0) / 100
        animate_projections = st.toggle("Animate Projections", True)
        
        # Chat and Email configuration
        st.subheader("Optional Add-ons")
        chat_sessions = st.slider(
            "Chat Sessions Volume", 
            min_value=10000, 
            max_value=50000, 
            value=50000, 
            step=1000
        )
        email_volume = st.slider(
            "Email Volume", 
            min_value=10000, 
            max_value=25000, 
            value=20000, 
            step=500
        )
        
        st.markdown("---")
        st.caption("ℹ️ *Outbound dialing requires customer-provided dialer")

    # Process data with add-ons
    processed_df = process_pricing_data(df, chat_sessions, email_volume, outbound_toggle)

    # Get outbound_toggle from Streamlit session state
    outbound_toggle = st.session_state.get('outbound_toggle', False)
    
    processed_df = clean_numeric_data(processed_df)
    #display_pricing_data(processed_df, outbound_toggle)
    #display_df = format_for_display(processed_df, outbound_toggle)
    #st.dataframe(display_df, use_container_width=True)  # Show formatted version

    filtered = processed_df[processed_df['Agents'] == selected_agent].copy()

    # Create detailed cost table
    detailed_table = create_detailed_cost_table(processed_df, selected_agent, chat_sessions, email_volume, outbound_toggle)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Cost Comparison", 
        "📝 Detailed Breakdown", 
        "⚖️ Breakeven Analysis",
        "📈 Multi-Year Projection",
        "📁 Raw Data"
    ])


    with tab1:
        # Cost Comparison tab content
        st.markdown(f"### 💵 {time_period} Cost Comparison{'*' if outbound_toggle else ''}")
        
        fixed_cost = filtered[filtered['Option'] == 'Fixed Pricing'][f"{time_period}Cost"].values[0]
        payg_cost = filtered[filtered['Option'] == 'Pay-As-You-Go'][f"{time_period}Cost"].values[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Fixed Pricing Cost",
                format_currency(fixed_cost, show_inr, usd_to_inr),
                help="Includes implementation cost amortization"
            )
        with col2:
            st.metric(
                "Pay-As-You-Go Cost",
                format_currency(payg_cost, show_inr, usd_to_inr),
                help="Pure operational costs"
            )
        
        # Visual comparison
        fig = px.bar(
            processed_df,
            x='Agents',
            y=f"{time_period}Cost",
            color='Option',
            barmode='group',
            title=f"{time_period} Cost Comparison{'*' if outbound_toggle else ''}",
            labels={f"{time_period}Cost": 'Cost (USD)' + (' / INR' if show_inr else '')}
        )
        
        # Add annotation if outbound is enabled
        if outbound_toggle:
            fig.update_layout(
                annotations=[
                    dict(
                        x=0.5,
                        y=-0.15,
                        xref="paper",
                        yref="paper",
                        text="* Includes 10% outbound dialing surcharge (customer-provided dialer)",
                        showarrow=False,
                        font=dict(size=10)
                    )
                ]
            )
        
        fig.add_vline(x=selected_agent-0.5, line_width=3, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Detailed Breakdown tab content
        st.markdown("### 📝 Detailed Cost Breakdown")
        
        # Show detailed table for selected option
        option = st.radio("Show details for:", ["Fixed Pricing", "Pay-As-You-Go"], horizontal=True)
        
        if option == "Fixed Pricing":
            cols = ['Metric', f'Fixed_{time_period}']
            display_df = detailed_table[cols].rename(columns={f'Fixed_{time_period}': time_period})
        else:
            cols = ['Metric', f'PAYG_{time_period}']
            display_df = detailed_table[cols].rename(columns={f'PAYG_{time_period}': time_period})
        
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True
        )
        
        # PDF Export
        st.markdown("### 📄 Export Detailed Report")
        config = {
            'agent_count': selected_agent,
            'time_period': time_period,
            'outbound': outbound_toggle,
            'chat_sessions': chat_sessions,
            'email_volume': email_volume
        }

        pdf = create_detailed_pdf(detailed_table, config, usd_to_inr)
        st.markdown(get_pdf_download_link(pdf, f"kcube_report_{selected_agent}_agents.pdf"), unsafe_allow_html=True)

    with tab3:
        # Breakeven Analysis tab content
        st.markdown(f"### ⚖️ Breakeven Analysis{'*' if outbound_toggle else ''}")
        
        compare_df = processed_df.pivot_table(
            index='Agents',
            columns='Option',
            values=f"{time_period}Cost",
            aggfunc='sum'
        ).reset_index()
        
        compare_df['Savings'] = compare_df['Fixed Pricing'] - compare_df['Pay-As-You-Go']
        compare_df['Breakeven Months'] = (15000 / compare_df['Savings']).round(1)
        
        current_breakeven = compare_df[compare_df['Agents'] == selected_agent]
        if not current_breakeven.empty:
            savings = current_breakeven['Savings'].values[0]
            breakeven_months = current_breakeven['Breakeven Months'].values[0]
            
            st.metric(
                "Months to Breakeven",
                f"{breakeven_months} months",
                delta=f"Saves {format_currency(savings, show_inr, usd_to_inr)} {time_period.lower()}",
                delta_color="inverse" if breakeven_months > 12 else "normal"
            )
            
            fig = px.line(
                compare_df,
                x='Agents',
                y='Breakeven Months',
                title=f"Months to Recover Implementation Cost{'*' if outbound_toggle else ''}",
                markers=True
            )
            
            # Add annotation if outbound is enabled
            if outbound_toggle:
                fig.update_layout(
                    annotations=[
                        dict(
                            x=0.5,
                            y=-0.15,
                            xref="paper",
                            yref="paper",
                            text="* Includes 10% outbound dialing surcharge (customer-provided dialer)",
                            showarrow=False,
                            font=dict(size=10)
                        )
                    ]
                )
            
            fig.add_hline(y=12, line_dash="dash", line_color="red", annotation_text="1 Year Benchmark")
            fig.add_vline(x=selected_agent-0.5, line_width=3, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown(f"### 📈 {analysis_years}-Year Cost Projection{'*' if outbound_toggle else ''}")

        # Calculate multi-year costs
        fixed_yearly = filtered[filtered['Option'] == 'Fixed Pricing']['YearlyCost'].values[0]
        payg_yearly = filtered[filtered['Option'] == 'Pay-As-You-Go']['YearlyCost'].values[0]

        years = list(range(1, analysis_years + 1))
        fixed_costs = calculate_multi_year_costs(fixed_yearly, analysis_years, growth_rate)
        payg_costs = calculate_multi_year_costs(payg_yearly, analysis_years, growth_rate)

        # Create figure with secondary y-axis
        fig = make_subplots(
            specs=[[{"secondary_y": True}]],
            subplot_titles=("Cumulative Costs", "Annual Costs")
        )

        # Add traces
        fig.add_trace(
            go.Scatter(
                x=years,
                y=np.cumsum(fixed_costs),
                name="Fixed Pricing (Cumulative)",
                line=dict(color='blue', width=4),
                visible=True,
                hovertemplate="Year %{x}<br>Total: $%{y:,.2f}"
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=years,
                y=np.cumsum(payg_costs),
                name="PayG (Cumulative)",
                line=dict(color='green', width=4),
                visible=True,
                hovertemplate="Year %{x}<br>Total: $%{y:,.2f}"
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Bar(
                x=years,
                y=fixed_costs,
                name="Fixed (Annual)",
                marker_color='lightblue',
                opacity=0.7,
                visible=True,
                hovertemplate="Year %{x}<br>Annual: $%{y:,.2f}"
            ),
            secondary_y=True
        )

        fig.add_trace(
            go.Bar(
                x=years,
                y=payg_costs,
                name="PayG (Annual)",
                marker_color='lightgreen',
                opacity=0.7,
                visible=True,
                hovertemplate="Year %{x}<br>Annual: $%{y:,.2f}"
            ),
            secondary_y=True
        )

        # Add implementation cost marker
        fig.add_hline(
            y=15000,
            line_dash="dot",
            line_color="red",
            annotation_text="Implementation Cost",
            annotation_position="bottom right",
            secondary_y=False
        )

        # Animation with smooth transitions
        if animate_projections:
            frames = [
                go.Frame(
                    data=[
                        go.Scatter(
                            x=years[:k+1],
                            y=np.cumsum(fixed_costs[:k+1]),
                            line=dict(color='blue', width=4)
                        ),
                        go.Scatter(
                            x=years[:k+1],
                            y=np.cumsum(payg_costs[:k+1]),
                            line=dict(color='green', width=4)
                        ),
                        go.Bar(
                            x=years[:k+1],
                            y=fixed_costs[:k+1],
                            marker_color='lightblue'
                        ),
                        go.Bar(
                            x=years[:k+1],
                            y=payg_costs[:k+1],
                            marker_color='lightgreen'
                        )
                    ],
                    name=f"frame_{k}"
                ) for k in range(analysis_years)
            ]

            fig.frames = frames

            # Configure animation settings
            animation_settings = {
                "frame": {
                    "duration": 800,  # Slower animation
                    "redraw": True,
                    "easing": "cubic-in-out"  # Smooth easing
                },
                "transition": {
                    "duration": 600,  # Smooth transition between frames
                    "easing": "cubic-in-out"
                },
                "mode": "immediate"
            }

            fig.update_layout(
                updatemenus=[{
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "▶️ Play",
                            "method": "animate",
                            "args": [
                                None, 
                                {
                                    "frame": animation_settings["frame"],
                                    "transition": animation_settings["transition"],
                                    "fromcurrent": True,
                                    "mode": animation_settings["mode"]
                                }
                            ]
                        },
                        {
                            "label": "⏸️ Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"
                                }
                            ]
                        },
                        {
                            "label": "⏩ Fast Forward",
                            "method": "animate",
                            "args": [
                                None, 
                                {
                                    "frame": {"duration": 300, "redraw": True},
                                    "transition": {"duration": 200},
                                    "fromcurrent": True,
                                    "mode": "immediate"
                                }
                            ]
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 10},
                    "showactive": True,
                    "x": 0.1,
                    "xanchor": "left",
                    "y": 1.2,
                    "yanchor": "top"
                }]
            )

        # Layout configuration with proper margins
        fig.update_layout(
            title={
                'text': f"{analysis_years}-Year Cost Projection (Growth: {growth_rate*100:.1f}%){'*' if outbound_toggle else ''}",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Years",
            yaxis_title="Cumulative Cost (USD)",
            yaxis2_title="Annual Cost (USD)",
            hovermode="x unified",
            showlegend=True,
            margin=dict(t=100, b=100),  # Add top/bottom margin
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Add outbound footnote with proper positioning
        if outbound_toggle:
            fig.add_annotation(
                x=0.5,
                y=-0.3,
                xref="paper",
                yref="paper",
                text="* Includes 10% outbound dialing surcharge (customer-provided dialer)",
                showarrow=False,
                font=dict(size=10),
                align="center"
            )

        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        # Raw Data tab content
        st.markdown("### 📁 Raw Data")
        st.dataframe(processed_df)
        
        csv = processed_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Processed Data",
            data=csv,
            file_name="kcube_pricing_analysis.csv",
            mime="text/csv"
        )
    
    # Recommendation
    st.markdown("### 🚀 Recommendation")
    if not filtered.empty:
        fixed_cost = filtered[filtered['Option'] == 'Fixed Pricing'][f"{time_period}Cost"].values[0]
        payg_cost = filtered[filtered['Option'] == 'Pay-As-You-Go'][f"{time_period}Cost"].values[0]
        
        if payg_cost < fixed_cost:
            savings = fixed_cost - payg_cost
            breakeven = (15000 / savings) if savings > 0 else 0
            st.success(f"""
            **Recommendation for {selected_agent} Agents:**
            - Choose **Pay-As-You-Go** 
            - Save **{format_currency(savings, show_inr, usd_to_inr)} {time_period.lower()}**
            - Breakeven in **{breakeven:.1f} months**
            """)
        else:
            savings = payg_cost - fixed_cost
            st.success(f"""
            **Recommendation for {selected_agent} Agents:**
            - Choose **Fixed Pricing** 
            - Save **{format_currency(savings, show_inr, usd_to_inr)} {time_period.lower()}**
            - Immediate savings
            """)

    # Footer
    st.markdown("---")
    st.caption(f"""
    **Note:**  
    *Outbound dialing: adds 10% to the base telephony cost (Customer must provide their own dialer)*  
    - Chat Agent Cost: ${BASE_CHAT_RATE:.3f} per session (Default: ${DEFAULT_CHAT_AGENT_COST:,.2f} for 50,000 sessions)  
    - Email Agent Cost: ${BASE_EMAIL_RATE:.3f} per email (Default: ${DEFAULT_EMAIL_AGENT_COST:,.2f} for 20,000 emails)  
    - All costs shown exclude taxes  
    - Implementation cost is $15,000 (one-time)  
    - Report generated on {date.today().strftime('%Y-%m-%d')}  
    """)

if __name__ == "__main__":
    main()