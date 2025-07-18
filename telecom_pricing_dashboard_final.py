from fpdf import FPDF, XPos, YPos, FontFace
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import re
from datetime import date
from scipy import interpolate
import base64
from io import BytesIO
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CHAT_COSTS = {1000: 240, 5000: 240, 10000: 480, 25000: 1200, 50000: 2400}
DEFAULT_EMAIL_COST = 1200  # For 20,000 emails
IMPLEMENTATION_COST = 15000  # One-time cost
MINUTES_PER_AGENT = 7200  # Monthly minutes per agent (120 hours)

# Helper Functions
def calculate_multi_year_costs(base_cost, years, growth_rate=0.0):
    return [base_cost * (1 + growth_rate)**year for year in range(years)]

def extract_cost(value):
    if pd.isna(value) or value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        num_str = re.sub(r'[^\d.-]', '', value)
        return float(num_str) if num_str else 0.0
    return 0.0

def format_currency(value, show_inr=False, exchange_rate=85.0):
    try:
        value = float(value)
        if show_inr:
            return f"${value:,.2f} (₹{value*exchange_rate:,.2f})"
        return f"${value:,.2f}"
    except (ValueError, TypeError):
        return "N/A"

def calculate_chat_cost(sessions):
    if sessions <= 1000: return DEFAULT_CHAT_COSTS[1000] * (sessions/1000)
    elif sessions <= 5000: return DEFAULT_CHAT_COSTS[5000] * (sessions/5000)
    elif sessions <= 10000: return DEFAULT_CHAT_COSTS[10000] * (sessions/10000)
    elif sessions <= 25000: return DEFAULT_CHAT_COSTS[25000] * (sessions/25000)
    else: return DEFAULT_CHAT_COSTS[50000] * (sessions/50000)

def calculate_email_cost(emails):
    return DEFAULT_EMAIL_COST * (emails/20000) if emails <= 20000 else DEFAULT_EMAIL_COST + (emails-20000)*0.06

def get_pdf_download_link(pdf, filename):
    try:
        pdf_bytes = BytesIO()
        pdf.output(pdf_bytes)
        pdf_bytes = pdf_bytes.getvalue()
        b64 = base64.b64encode(pdf_bytes).decode()
        return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF Report</a>'
    except Exception as e:
        logger.error(f"PDF link generation failed: {str(e)}")
        return "<p style='color:red'>PDF generation failed</p>"

def generate_pdf_report(config, numeric_table, display_table, recommendation, notes):
    """Generate professional PDF report with dynamic currency display"""
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Set professional styling
        pdf.set_font("helvetica", "", 10)
        line_height = 6
        exchange_rate = config.get('exchange_rate', 87.0)
        show_inr = config.get('show_inr', False)
        
        # --- Header ---
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, "Eva GenAI Platform - Pricing Analysis", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(0, line_height, f"Prepared for: {config.get('client_name', 'Client')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("helvetica", "", 10)
        pdf.cell(0, line_height, f"Report Date: {date.today().strftime('%B %d, %Y')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if show_inr:
            pdf.cell(0, line_height, f"Conversion Rate: 1 USD = INR {exchange_rate:.2f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # --- Executive Summary ---
        pdf.ln(5)
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, line_height, "Executive Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # Get costs
        period = config['time_period']
        fixed_cost = numeric_table[numeric_table['Metric'] == 'Total Cost (Excl. Implementation)'][f'Fixed_{period}_Value'].values[0]
        payg_cost = numeric_table[numeric_table['Metric'] == 'Total Cost (Excl. Implementation)'][f'PAYG_{period}_Value'].values[0]
        
        # Format recommendation
        if payg_cost < fixed_cost:
            savings = fixed_cost - payg_cost
            rec_text = f"Recommended: Pay-As-You-Go (Saves ${savings:,.2f})"
            if show_inr:
                rec_text += f" or INR {savings*exchange_rate:,.2f}"
        else:
            savings = payg_cost - fixed_cost
            rec_text = f"Recommended: Fixed Pricing (Saves ${savings:,.2f})"
            if show_inr:
                rec_text += f" or INR {savings*exchange_rate:,.2f}"
        
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(0, line_height, f"Optimal Pricing Model for {config['agent_count']} Agents:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("helvetica", "", 10)
        pdf.cell(0, line_height, rec_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        impl_cost_text = f"Implementation Cost: $15,000.00"
        if show_inr:
            impl_cost_text += f" or INR {15000*exchange_rate:,.2f}"
        impl_cost_text += " (one-time)"
        pdf.cell(0, line_height, impl_cost_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # Divider
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)
        
        # --- Agent Metrics ---
        pdf.set_font("helvetica", "", 10)
        pdf.cell(40, line_height, "Agent's:", border=0)
        pdf.cell(20, line_height, str(config['agent_count']), border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        pdf.cell(40, line_height, "Total Minutes:", border=0)
        pdf.cell(20, line_height, f"{config['agent_count'] * config['minutes_per_agent']:.0f}", border=0, 
                new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        pdf.cell(40, line_height, "Total Hours:", border=0)
        pdf.cell(20, line_height, f"{(config['agent_count'] * config['minutes_per_agent'])/60:.0f}", border=0,
                new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        pdf.ln(5)
        
        # --- Detailed Cost Comparison ---
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, line_height, "Detailed Cost Comparison", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # Define table structure based on INR setting
        if show_inr:
            col_widths = [60, 25, 25, 25, 25]  # Component, Fixed USD, Fixed INR, PayG USD, PayG INR
        else:
            col_widths = [80, 40, 40]  # Component, Fixed USD, PayG USD
        
        pdf.set_fill_color(200, 220, 255)
        pdf.set_font("helvetica", "B", 9)
        
        # Header rows
        if show_inr:
            # Header row 1 (INR ON)
            pdf.cell(col_widths[0], line_height, "Cost Component", border=1, fill=True)
            pdf.cell(col_widths[1]+col_widths[2], line_height, "Fixed", border=1, fill=True, align='C')
            pdf.cell(col_widths[3]+col_widths[4], line_height, "PayG", border=1, fill=True, align='C', 
                    new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            
            # Header row 2 (INR ON)
            pdf.cell(col_widths[0], line_height, "", border=1, fill=True)
            pdf.cell(col_widths[1], line_height, "USD", border=1, fill=True, align='C')
            pdf.cell(col_widths[2], line_height, "INR", border=1, fill=True, align='C')
            pdf.cell(col_widths[3], line_height, "USD", border=1, fill=True, align='C')
            pdf.cell(col_widths[4], line_height, "INR", border=1, fill=True, align='C', 
                    new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            # Single header row (INR OFF)
            pdf.cell(col_widths[0], line_height, "Cost Component", border=1, fill=True)
            pdf.cell(col_widths[1], line_height, "Fixed (USD)", border=1, fill=True, align='C')
            pdf.cell(col_widths[2], line_height, "PayG (USD)", border=1, fill=True, align='C', 
                    new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # Table content
        cost_components = [
            "In/Out Bound Telephony Cost *",
            "AI Costs",
            "Tech Infra",
            "Operational & Support Cost",
            "Implementation Cost",
            "Chat Agent",
            "Email Agent",
            "Total Cost (Excl. Implementation)"
        ]
        
        pdf.set_font("helvetica", "", 8)
        for comp in cost_components:
            # Get values from numeric table
            row = numeric_table[numeric_table['Metric'] == comp]
            
            # Fixed values
            if not row.empty:
                fixed_val = row[f'Fixed_{period}_Value'].values[0]
                if pd.isna(fixed_val):
                    fixed_usd = fixed_inr = "Included"
                else:
                    fixed_usd = f"${fixed_val:,.2f}" if fixed_val != 0 else "-"
                    fixed_inr = f"INR {fixed_val*exchange_rate:,.2f}" if fixed_val != 0 else "-"
            else:
                fixed_usd = fixed_inr = "Included" if comp in ["AI Costs", "Tech Infra", "Operational & Support Cost"] else "-"
            
            # PayG values
            if not row.empty:
                payg_val = row[f'PAYG_{period}_Value'].values[0]
                if pd.isna(payg_val):
                    payg_usd = payg_inr = "Included"
                else:
                    payg_usd = f"${payg_val:,.2f}" if payg_val != 0 else "-"
                    payg_inr = f"INR {payg_val*exchange_rate:,.2f}" if payg_val != 0 else "-"
            else:
                payg_usd = payg_inr = "Included" if comp in ["AI Costs", "Tech Infra", "Operational & Support Cost"] else "-"
            
            # Special cases
            if comp == "Implementation Cost":
                fixed_usd = payg_usd = "$15,000.00"
                fixed_inr = payg_inr = f"INR {15000*exchange_rate:,.2f}"
            
            # Draw row
            pdf.cell(col_widths[0], line_height, comp, border=1)
            
            if show_inr:
                pdf.cell(col_widths[1], line_height, fixed_usd, border=1, align='C')
                pdf.cell(col_widths[2], line_height, fixed_inr, border=1, align='C')
                pdf.cell(col_widths[3], line_height, payg_usd, border=1, align='C')
                pdf.cell(col_widths[4], line_height, payg_inr, border=1, align='C', 
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                pdf.cell(col_widths[1], line_height, fixed_usd, border=1, align='C')
                pdf.cell(col_widths[2], line_height, payg_usd, border=1, align='C', 
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # Footer note
        pdf.set_font("helvetica", "I", 8)
        pdf.cell(0, line_height, "* Outbound dialing adds 10% to base telephony cost", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # --- Cost Comparison ---
        pdf.ln(5)
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, line_height, "Cost Comparison", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("helvetica", "", 10)
        
        if show_inr:
            pdf.cell(0, line_height, f"Fixed: ${fixed_cost:,.2f} or INR {fixed_cost*exchange_rate:,.2f}", 
                    new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, line_height, f"PayG: ${payg_cost:,.2f} or INR {payg_cost*exchange_rate:,.2f}", 
                    new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            pdf.cell(0, line_height, f"Fixed: ${fixed_cost:,.2f}", 
                    new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, line_height, f"PayG: ${payg_cost:,.2f}", 
                    new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # --- Recommendation ---
        pdf.ln(2)
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, line_height, "Recommendation", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("helvetica", "", 10)
        for line in recommendation.split('\n'):
            pdf.multi_cell(0, line_height, line.strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # --- Notes ---
        pdf.ln(2)
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, line_height, "Notes", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("helvetica", "", 9)
        notes_lines = [
            "* Outbound dialing adds 10% to base telephony cost (Customer must provide their own dialer)",
            "- Chat Agent Cost: Tiered pricing (1000: $240, 5000: $240, 10000: $480, 25000: $1200, 50000: $2400)",
            "- Email Agent Cost: $1200 for 20000 emails ($0.06 per additional email)",
            "- Implementation cost is $15000 (one-time)",
            f"- Standard agent time: {config['minutes_per_agent']} minutes/month ({(config['minutes_per_agent']/60):.1f} hours)",
            f"- Report generated on {date.today().strftime('%Y-%m-%d')}"
        ]
        for note in notes_lines:
            pdf.multi_cell(0, line_height, note, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        return pdf
        
    except Exception as e:
        logger.error(f"PDF generation failed: {str(e)}")
        st.error(f"Failed to generate PDF report: {str(e)}")
        return None
        
def generate_excel_report(config, numeric_table, display_table, recommendation, notes):
    """Generate an Excel report with all pricing details"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Save display table
        display_table.to_excel(writer, sheet_name='Cost Breakdown', index=False)
        
        # Save numeric table
        numeric_table.to_excel(writer, sheet_name='Raw Data', index=False)
        
        # Add summary sheet
        workbook = writer.book
        summary_sheet = workbook.add_worksheet('Summary')
        
        # Write configuration
        summary_sheet.write(0, 0, 'Configuration Parameters')
        summary_sheet.write(1, 0, 'Agent Count')
        summary_sheet.write(1, 1, config['agent_count'])
        summary_sheet.write(2, 0, 'Time Period')
        summary_sheet.write(2, 1, config['time_period'])
        summary_sheet.write(3, 0, 'Minutes per Agent')
        summary_sheet.write(3, 1, f"{config['minutes_per_agent']} ({(config['minutes_per_agent']/60):.1f} hours)")
        summary_sheet.write(4, 0, 'Outbound Telephony')
        summary_sheet.write(4, 1, 'Yes (+10%)' if config['outbound'] else 'No')
        summary_sheet.write(5, 0, 'Chat Sessions')
        summary_sheet.write(5, 1, config['chat_sessions'] if config['chat_sessions'] > 0 else 'Disabled')
        summary_sheet.write(6, 0, 'Email Volume')
        summary_sheet.write(6, 1, config['email_volume'] if config['email_volume'] > 0 else 'Disabled')
        summary_sheet.write(7, 0, 'Analysis Years')
        summary_sheet.write(7, 1, config['analysis_years'])
        summary_sheet.write(8, 0, 'Growth Rate')
        summary_sheet.write(8, 1, f"{config['growth_rate']*100:.1f}%")
        summary_sheet.write(9, 0, 'Client Name')
        summary_sheet.write(9, 1, config.get('client_name', 'Client'))
        
        # Write recommendation
        summary_sheet.write(11, 0, 'Recommendation')
        summary_sheet.write(12, 0, recommendation.replace('\n', ' '))
        
        # Write notes
        summary_sheet.write(14, 0, 'Notes')
        summary_sheet.write(15, 0, notes.replace('\n', ' '))
    
    return output.getvalue()

def get_report_download_links(config, numeric_table, display_table, fixed_cost, payg_cost, time_period):
    """Generate both PDF and Excel report download links"""
    # Generate recommendation
    if payg_cost < fixed_cost:
        savings = fixed_cost - payg_cost
        breakeven = (IMPLEMENTATION_COST / savings) if savings > 0 else 0
        recommendation = f"""
        Recommendation for {config['agent_count']} Agents:
        - Choose Pay-As-You-Go 
        - Save ${savings:,.2f} {time_period.lower()}
        - Breakeven in {breakeven:.1f} months
        """
    else:
        savings = payg_cost - fixed_cost
        recommendation = f"""
        Recommendation for {config['agent_count']} Agents:
        - Choose Fixed Pricing 
        - Save ${savings:,.2f} {time_period.lower()}
        - Immediate savings
        """
    
    # Generate notes with proper formatting
    notes = f"""
    *Outbound dialing: adds 10% to the base telephony cost (Customer must provide their own dialer)
    
    - Chat Agent Cost: Tiered pricing (1K: $240, 5K: $240, 10K: $480, 25K: $1,200, 50K: $2,400)
    
    - Email Agent Cost: $1,200 for 20,000 emails ($0.06 per additional email)
    
    - Implementation cost: $15,000 (one-time)
    
    - Standard agent time: {config['minutes_per_agent']} minutes/month ({(config['minutes_per_agent']/60):.1f} hours)
    
    Report generated on {date.today().strftime('%Y-%m-%d')}
    """
    
    # Generate PDF
    pdf_report = generate_pdf_report(config, numeric_table, display_table, recommendation, notes)
    
    # Generate Excel
    excel_data = generate_excel_report(config, numeric_table, display_table, recommendation, notes)
    
    # Create download links
    if pdf_report:
        pdf_link = get_pdf_download_link(pdf_report, f"kcube_report_{config['agent_count']}_agents.pdf")
    else:
        pdf_link = "<p style='color:red'>PDF generation failed</p>"
    
    if excel_data:
        b64 = base64.b64encode(excel_data).decode()
        excel_link = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="kcube_report_{config["agent_count"]}_agents.xlsx">Download Excel Report</a>'
    else:
        excel_link = "<p style='color:red'>Excel generation failed</p>"
    
    return pdf_link, excel_link
    
# Data Processing
@st.cache_data
def load_excel_data(uploaded_file):
    try:
        xls = pd.ExcelFile(uploaded_file)
        required_sheets = ['Fixed', 'PayG']
        missing_sheets = [sheet for sheet in required_sheets if sheet not in xls.sheet_names]
        if missing_sheets:
            raise ValueError(f"Missing required sheets: {missing_sheets}")

        def process_fixed_sheet(df):
            original_agents = [1,5,10,25,50]
            results = []
            for _, row in df.iterrows():
                metric = str(row.iloc[0]).strip()
                if 'total cost' in metric.lower():
                    monthly_costs = [extract_cost(row.iloc[i*2+1]) for i in range(len(original_agents))]
                    yearly_costs = [extract_cost(row.iloc[i*2+2]) for i in range(len(original_agents))]
                    
                    monthly_interp = interpolate.interp1d(original_agents, monthly_costs, kind='linear', fill_value='extrapolate')
                    yearly_interp = interpolate.interp1d(original_agents, yearly_costs, kind='linear', fill_value='extrapolate')
                    
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
                            'Option': 'Fixed Pricing'
                        })
                    break
            return pd.DataFrame(results)

        def process_payg_sheet(df):
            payg_rates = {
                'Agents': [10, 20, 30],
                'MonthlyCost': [extract_cost(df.iloc[6,1]), extract_cost(df.iloc[6,2]), extract_cost(df.iloc[6,3])]
            }
            
            monthly_interp = interpolate.interp1d(payg_rates['Agents'], payg_rates['MonthlyCost'], kind='linear', fill_value='extrapolate')
            
            all_agents = [1,5,10,15,20,25,30,40,50]
            results = []
            for agent in all_agents:
                monthly = float(monthly_interp(agent))
                yearly = monthly * 12
                
                results.append({
                    'Agents': agent,
                    'MonthlyCost': monthly,
                    'YearlyCost': yearly,
                    'Option': 'Pay-As-You-Go'
                })
            
            return pd.DataFrame(results)

        df_fixed = pd.read_excel(xls, sheet_name='Fixed', header=None)
        df_fixed_processed = process_fixed_sheet(df_fixed)
        
        df_payg = pd.read_excel(xls, sheet_name='PayG', header=None)
        df_payg_processed = process_payg_sheet(df_payg)
        
        return pd.concat([df_fixed_processed, df_payg_processed])
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return pd.DataFrame()

def process_pricing_data(df, chat_sessions, email_volume, outbound_toggle, minutes_per_agent):
    if df.empty:
        return df
        
    for col in ['MonthlyCost', 'YearlyCost']:
        if col in df.columns:
            df[col] = df[col].apply(extract_cost)
    
    # Adjust costs based on actual minutes vs standard minutes (7200)
    minute_adjustment = minutes_per_agent / MINUTES_PER_AGENT
    telephony_multiplier = 1.1 if outbound_toggle else 1.0
    
    df['MonthlyCost'] = df['MonthlyCost'] * telephony_multiplier * minute_adjustment
    df['YearlyCost'] = df['MonthlyCost'] * 12
    
    if chat_sessions > 0:
        chat_cost = calculate_chat_cost(chat_sessions)
        df['MonthlyCost'] += chat_cost
        df['YearlyCost'] += chat_cost * 12
    
    if email_volume > 0:
        email_cost = calculate_email_cost(email_volume)
        df['MonthlyCost'] += email_cost
        df['YearlyCost'] += email_cost * 12
    
    return df

def create_detailed_cost_table(df, agent_count, chat_sessions, email_volume, outbound_toggle, minutes_per_agent):
    fixed = df[(df['Agents'] == agent_count) & (df['Option'] == 'Fixed Pricing')]
    payg = df[(df['Agents'] == agent_count) & (df['Option'] == 'Pay-As-You-Go')]
    
    metrics = [
        "Agent's",
        "Total Minutes",
        "Total Hours",
        "In/Out Bound Telephony Cost *",
        "AI Costs",
        "Tech Infra",
        "Operational & Support Cost",
        "Implementation Cost",
        "Chat Agent",
        "Email Agent",
        "Total Cost (Excl. Implementation)"
    ]
    
    # Calculate values first (numeric)
    fixed_monthly_cost = float(fixed['MonthlyCost'].values[0]) if not fixed.empty else 0
    payg_monthly_cost = float(payg['MonthlyCost'].values[0]) if not payg.empty else 0
    
    # Create numeric dataframe for calculations
    numeric_data = {
        'Metric': metrics,
        'Fixed_Monthly_Value': [
            fixed['Agents'].values[0] if not fixed.empty else 0,
            minutes_per_agent * (fixed['Agents'].values[0] if not fixed.empty else 0),
            round(minutes_per_agent * (fixed['Agents'].values[0] if not fixed.empty else 0) / 60, 1),
            fixed_monthly_cost * 0.9,
            np.nan,  # For "Included" items
            np.nan,
            np.nan,
            15000,
            calculate_chat_cost(chat_sessions) if chat_sessions > 0 else 0,
            calculate_email_cost(email_volume) if email_volume > 0 else 0,
            fixed_monthly_cost
        ],
        'PAYG_Monthly_Value': [
            payg['Agents'].values[0] if not payg.empty else 0,
            minutes_per_agent * (payg['Agents'].values[0] if not payg.empty else 0),
            round(minutes_per_agent * (payg['Agents'].values[0] if not payg.empty else 0) / 60, 1),
            payg_monthly_cost * 0.9,
            np.nan,
            np.nan,
            np.nan,
            15000,
            calculate_chat_cost(chat_sessions) if chat_sessions > 0 else 0,
            calculate_email_cost(email_volume) if email_volume > 0 else 0,
            payg_monthly_cost
        ]
    }
    
    # Create display dataframe (with formatted strings)
    display_data = {
        'Metric': metrics,
        'Fixed_Monthly': [
            str(fixed['Agents'].values[0]) if not fixed.empty else "0",
            f"{minutes_per_agent * (fixed['Agents'].values[0] if not fixed.empty else 0):,.0f}",
            f"{round(minutes_per_agent * (fixed['Agents'].values[0] if not fixed.empty else 0) / 60, 1):,.1f}",
            f"${fixed_monthly_cost * 0.9:,.2f}{'*' if outbound_toggle else ''}",
            "Included",
            "Included",
            "Included",
            "$15,000 (One-time)",
            f"${calculate_chat_cost(chat_sessions):,.2f}" if chat_sessions > 0 else "Not enabled",
            f"${calculate_email_cost(email_volume):,.2f}" if email_volume > 0 else "Not enabled",
            f"${fixed_monthly_cost:,.2f}" if fixed_monthly_cost else "N/A"
        ],
        'PAYG_Monthly': [
            str(payg['Agents'].values[0]) if not payg.empty else "0",
            f"{minutes_per_agent * (payg['Agents'].values[0] if not payg.empty else 0):,.0f}",
            f"{round(minutes_per_agent * (payg['Agents'].values[0] if not payg.empty else 0) / 60, 1):,.1f}",
            f"${payg_monthly_cost * 0.9:,.2f}{'*' if outbound_toggle else ''}",
            "Included",
            "Included",
            "Included",
            "$15,000 (One-time)",
            f"${calculate_chat_cost(chat_sessions):,.2f}" if chat_sessions > 0 else "Not enabled",
            f"${calculate_email_cost(email_volume):,.2f}" if email_volume > 0 else "Not enabled",
            f"${payg_monthly_cost:,.2f}" if payg_monthly_cost else "N/A"
        ],
        'Fixed_Yearly': [
            str(fixed['Agents'].values[0]) if not fixed.empty else "0",
            f"{minutes_per_agent * (fixed['Agents'].values[0] if not fixed.empty else 0) * 12:,.0f}",
            f"{round(minutes_per_agent * (fixed['Agents'].values[0] if not fixed.empty else 0) * 12 / 60, 1):,.1f}",
            f"${fixed_monthly_cost * 0.9 * 12:,.2f}{'*' if outbound_toggle else ''}",
            "Included",
            "Included",
            "Included",
            "$15,000 (One-time)",
            f"${calculate_chat_cost(chat_sessions) * 12:,.2f}" if chat_sessions > 0 else "Not enabled",
            f"${calculate_email_cost(email_volume) * 12:,.2f}" if email_volume > 0 else "Not enabled",
            f"${fixed_monthly_cost * 12:,.2f}" if fixed_monthly_cost else "N/A"
        ],
        'PAYG_Yearly': [
            str(payg['Agents'].values[0]) if not payg.empty else "0",
            f"{minutes_per_agent * (payg['Agents'].values[0] if not payg.empty else 0) * 12:,.0f}",
            f"{round(minutes_per_agent * (payg['Agents'].values[0] if not payg.empty else 0) * 12 / 60, 1):,.1f}",
            f"${payg_monthly_cost * 0.9 * 12:,.2f}{'*' if outbound_toggle else ''}",
            "Included",
            "Included",
            "Included",
            "$15,000 (One-time)",
            f"${calculate_chat_cost(chat_sessions) * 12:,.2f}" if chat_sessions > 0 else "Not enabled",
            f"${calculate_email_cost(email_volume) * 12:,.2f}" if email_volume > 0 else "Not enabled",
            f"${payg_monthly_cost * 12:,.2f}" if payg_monthly_cost else "N/A"
        ]
    }
    
    return pd.DataFrame(numeric_data), pd.DataFrame(display_data)

# Main Dashboard
def main():
    # Configure page
    st.set_page_config(page_title="Kcube Pricing Dashboard", page_icon="📊", layout="wide")
    st.title("Eva GenAI Platform  - Pricing Options (Fixed & PayG)")
    
    uploaded_file = st.file_uploader("Upload Pricing Excel File", type=["xlsx"])
    if not uploaded_file:
        st.warning("Please upload your pricing Excel file")
        return

    df = load_excel_data(uploaded_file)
    if df.empty:
        st.error("Failed to load data. Please check the file format and structure.")
        return

    with st.sidebar:
        st.header("⚙️ Controls")
        client_name = st.text_input("Client Name", "Client")
        usd_to_inr = st.number_input("USD to INR Exchange Rate", min_value=1.0, value=85.0, step=0.5)
        show_inr = st.toggle("Show INR Pricing", value=False)
        
        agent_options = [1,5,10,15,20,25,30,40,50]
        selected_agent = st.selectbox("Select Agent Count", options=agent_options, index=2)
        
        # Minutes per agent slider
        minutes_per_agent = st.slider(
            "Minutes per Agent per Month",
            min_value=1000,
            max_value=10000,
            value=7200,  # Default 120 hours
            step=100
        )
        
        outbound_toggle = st.toggle("Include Outbound (+10%)*", True)
        time_period = st.radio("Time Period", ["Monthly", "Yearly"], index=0)
        
        st.subheader("Multi-Year Analysis")
        analysis_years = st.slider("Years to Analyze", 1, 5, 1)
        growth_rate = st.slider("Annual Growth Rate (%)", 0.0, 20.0, 5.0) / 100
        animate_projections = st.toggle("Animate Projections", True)
        
        st.subheader("Optional Add-ons")
        chat_enabled = st.toggle("Enable Chat Agent", False)
        if chat_enabled:
            chat_defaults = {1:1000, 5:5000, 10:10000, 25:25000, 50:50000}
            default_chat = chat_defaults.get(selected_agent, 1000)
            chat_sessions = st.selectbox(
                "Chat Sessions Volume", 
                options=[1000, 5000, 10000, 25000, 50000],
                index=[1000, 5000, 10000, 25000, 50000].index(default_chat)
            )
        else:
            chat_sessions = 0
        
        email_enabled = st.toggle("Enable Email Agent", False)
        if email_enabled:
            email_volume = st.slider("Email Volume", min_value=1000, max_value=50000, value=20000, step=500)
        else:
            email_volume = 0
        
        st.markdown("---")
        st.caption("ℹ️ *Outbound dialing requires customer-provided dialer")

    # Process data
    processed_df = process_pricing_data(df, chat_sessions, email_volume, outbound_toggle, minutes_per_agent)
    numeric_table, display_table = create_detailed_cost_table(
        processed_df, selected_agent, chat_sessions, email_volume, 
        outbound_toggle, minutes_per_agent
    )

    # Prepare configuration for reports
    config = {
        'agent_count': selected_agent,
        'time_period': time_period,
        'minutes_per_agent': minutes_per_agent,
        'outbound': outbound_toggle,
        'chat_sessions': chat_sessions,
        'email_volume': email_volume,
        'analysis_years': analysis_years,
        'growth_rate': growth_rate,
        'inr_rate': float(usd_to_inr),
        'show_inr': bool(show_inr),
        'client_name': client_name
    }

    # Get costs for selected agent
    try:
        fixed_cost = processed_df[(processed_df['Agents'] == selected_agent) & 
                                (processed_df['Option'] == 'Fixed Pricing')][f"{time_period}Cost"].values[0]
        payg_cost = processed_df[(processed_df['Agents'] == selected_agent) & 
                               (processed_df['Option'] == 'Pay-As-You-Go')][f"{time_period}Cost"].values[0]
        
        if payg_cost < fixed_cost:
            savings = fixed_cost - payg_cost
            breakeven = (IMPLEMENTATION_COST / savings) if savings > 0 else 0
            recommendation = f"""
            Recommendation for {selected_agent} Agents:
            - Choose Pay-As-You-Go 
            - Save ${savings:,.2f} {time_period.lower()}
            - Breakeven in {breakeven:.1f} months
            """
        else:
            savings = payg_cost - fixed_cost
            recommendation = f"""
            Recommendation for {selected_agent} Agents:
            - Choose Fixed Pricing 
            - Save ${savings:,.2f} {time_period.lower()}
            - Immediate savings
            """
    except Exception as e:
        st.error(f"Error generating recommendation: {str(e)}")
        recommendation = "Could not generate recommendation due to data issues"
        fixed_cost = 0
        payg_cost = 0

    # Generate notes
    notes = f"""
    *Outbound dialing: adds 10% to the base telephony cost (Customer must provide their own dialer)
    - Chat Agent Cost: Tiered pricing (1K: $240, 5K: $240, 10K: $480, 25K: $1,200, 50K: $2,400)
    - Email Agent Cost: $1,200 for 20,000 emails ($0.06 per additional email)
    - Implementation cost: $15,000 (one-time)
    - Standard agent time: {minutes_per_agent} minutes/month ({(minutes_per_agent/60):.1f} hours)
    """

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Cost Comparison", "📝 Detailed Breakdown", "⚖️ Breakeven Analysis", "📈 Multi-Year Projection", "📁 Raw Data"])

    with tab1:
        st.markdown(f"### 💵 {time_period} Cost Comparison{'*' if outbound_toggle else ''}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fixed Pricing Cost", format_currency(fixed_cost, show_inr, usd_to_inr))
        with col2:
            st.metric("Pay-As-You-Go Cost", format_currency(payg_cost, show_inr, usd_to_inr))
        
        cost_comparison_fig = px.bar(
            processed_df,
            x='Agents',
            y=f"{time_period}Cost",
            color='Option',
            barmode='group',
            title=f"{time_period} Cost Comparison{'*' if outbound_toggle else ''}"
        )
        cost_comparison_fig.add_vline(x=selected_agent-0.5, line_width=3, line_dash="dash", line_color="red")
        st.plotly_chart(cost_comparison_fig, use_container_width=True)
        
        # Time Metrics
        total_minutes = selected_agent * minutes_per_agent
        total_hours = total_minutes / 60
        st.markdown("### ⏱️ Time Calculation")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Minutes", f"{total_minutes:,.0f}")
            st.metric("Total Hours", f"{total_hours:,.1f}")
        with col2:
            st.metric("Minutes per Agent", f"{minutes_per_agent:,.0f}")
            st.metric("Hours per Agent", f"{minutes_per_agent/60:,.1f}")

    with tab2:
        st.markdown("### 📝 Detailed Cost Breakdown")
        option = st.radio("Show details for:", ["Fixed Pricing", "Pay-As-You-Go"], horizontal=True)
        
        if option == "Fixed Pricing":
            cols = ['Metric', f'Fixed_{time_period}']
            display_df = display_table[cols].rename(columns={f'Fixed_{time_period}': time_period})
        else:
            cols = ['Metric', f'PAYG_{time_period}']
            display_df = display_table[cols].rename(columns={f'PAYG_{time_period}': time_period})
        
        st.dataframe(display_df, hide_index=True, use_container_width=True)
        
        st.markdown("### 📄 Export Detailed Report")
        pdf_link, excel_link = get_report_download_links(
            config, numeric_table, display_table, 
            fixed_cost, payg_cost, time_period
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(pdf_link, unsafe_allow_html=True)
        with col2:
            st.markdown(excel_link, unsafe_allow_html=True)

    with tab3:
        st.markdown(f"### ⚖️ Breakeven Analysis{'*' if outbound_toggle else ''}")
        
        try:
            compare_df = processed_df.pivot_table(
                index='Agents',
                columns='Option',
                values=f"{time_period}Cost",
                aggfunc='sum'
            ).reset_index()
            compare_df['Savings'] = compare_df['Fixed Pricing'] - compare_df['Pay-As-You-Go']
            compare_df['Breakeven Months'] = (IMPLEMENTATION_COST / compare_df['Savings']).round(1)
            
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
        except Exception as e:
            st.error(f"Could not generate breakeven analysis: {str(e)}")

    with tab4:
        st.markdown(f"### 📈 {analysis_years}-Year Cost Projection{'*' if outbound_toggle else ''}")

        try:
            fixed_yearly = processed_df[(processed_df['Agents'] == selected_agent) & 
                                      (processed_df['Option'] == 'Fixed Pricing')]['YearlyCost'].values[0]
            payg_yearly = processed_df[(processed_df['Agents'] == selected_agent) & 
                                     (processed_df['Option'] == 'Pay-As-You-Go')]['YearlyCost'].values[0]

            years = list(range(1, analysis_years + 1))
            fixed_costs = calculate_multi_year_costs(fixed_yearly, analysis_years, growth_rate)
            payg_costs = calculate_multi_year_costs(payg_yearly, analysis_years, growth_rate)

            fig = make_subplots(specs=[[{"secondary_y": True}]], 
                              subplot_titles=("Cumulative Costs", "Annual Costs"))

            # Cumulative costs
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=np.cumsum(fixed_costs),
                    name="Fixed Pricing (Cumulative)",
                    line=dict(color='blue', width=4),
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
                    hovertemplate="Year %{x}<br>Total: $%{y:,.2f}"
                ),
                secondary_y=False
            )

            # Annual costs
            fig.add_trace(
                go.Bar(
                    x=years,
                    y=fixed_costs,
                    name="Fixed (Annual)",
                    marker_color='lightblue',
                    opacity=0.7,
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
                    hovertemplate="Year %{x}<br>Annual: $%{y:,.2f}"
                ),
                secondary_y=True
            )

            fig.add_hline(
                y=IMPLEMENTATION_COST,
                line_dash="dot",
                line_color="red",
                annotation_text="Implementation Cost",
                annotation_position="bottom right",
                secondary_y=False
            )

            if animate_projections:
                frames = [
                    go.Frame(
                        data=[
                            go.Scatter(x=years[:k+1], y=np.cumsum(fixed_costs[:k+1])),
                            go.Scatter(x=years[:k+1], y=np.cumsum(payg_costs[:k+1])),
                            go.Bar(x=years[:k+1], y=fixed_costs[:k+1]),
                            go.Bar(x=years[:k+1], y=payg_costs[:k+1])
                        ],
                        name=f"frame_{k}"
                    ) for k in range(analysis_years)
                ]

                fig.frames = frames
                fig.update_layout(
                    updatemenus=[{
                        "type": "buttons",
                        "buttons": [
                            {
                                "label": "▶️ Play",
                                "method": "animate",
                                "args": [None, {"frame": {"duration": 800, "redraw": True}}]
                            },
                            {
                                "label": "⏸️ Pause",
                                "method": "animate",
                                "args": [[None], {"frame": {"duration": 0, "redraw": False}}]
                            }
                        ]
                    }]
                )

            fig.update_layout(
                title=f"{analysis_years}-Year Cost Projection (Growth: {growth_rate*100:.1f}%){'*' if outbound_toggle else ''}",
                xaxis_title="Years",
                yaxis_title="Cumulative Cost (USD)",
                yaxis2_title="Annual Cost (USD)",
                hovermode="x unified"
            )

            if outbound_toggle:
                fig.add_annotation(
                    x=0.5,
                    y=-0.15,
                    xref="paper",
                    yref="paper",
                    text="* Includes 10% outbound dialing surcharge (customer-provided dialer)",
                    showarrow=False,
                    font=dict(size=10)
                )

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate projection: {str(e)}")

    with tab5:
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
    st.success(recommendation)

    # Footer
    st.markdown("---")
    st.caption(f"""
    **Note:**  
    *Outbound dialing: adds 10% to the base telephony cost (Customer must provide their own dialer)*  
    - Chat Agent Cost: Tiered pricing (1K: $240, 5K: $240, 10K: $480, 25K: $1,200, 50K: $2,400)  
    - Email Agent Cost: $1,200 for 20,000 emails ($0.06 per additional email)  
    - Implementation cost: $15,000 (one-time)  
    - Standard agent time: {minutes_per_agent} minutes/month ({(minutes_per_agent/60):.1f} hours)  
    - Report generated on {date.today().strftime('%Y-%m-%d')}  
    """)

if __name__ == "__main__":
    main()