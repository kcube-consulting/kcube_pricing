import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
from datetime import date
from scipy import interpolate
from fpdf import FPDF
import base64
from io import BytesIO
import os
import logging
from playwright.sync_api import sync_playwright
import subprocess
import sys
import html
import asyncio

# Fix for Windows asyncio issues
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configure Playwright paths
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '0'
os.environ['PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH'] = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
os.environ['DEBUG'] = 'pw:api'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CHAT_COSTS = {1000: 240, 5000: 240, 10000: 480, 25000: 1200, 50000: 2400}
DEFAULT_EMAIL_COST = 1200  # For 20,000 emails
IMPLEMENTATION_COST = 15000  # One-time cost
MINUTES_PER_AGENT = 7200  # Monthly minutes per agent (120 hours)

# Initialize session state for browser selection
if 'selected_browser' not in st.session_state:
    st.session_state.selected_browser = None
if 'browser_configured' not in st.session_state:
    st.session_state.browser_configured = False

# Configure browser settings
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '0'  # Use system browsers

# Browser Management Functions
def detect_available_browsers():
    """Check which browsers are available via Playwright"""
    available = []
    try:
        with sync_playwright() as p:
            for browser_type in [p.chromium, p.firefox, p.webkit]:
                try:
                    browser = browser_type.launch(headless=True)
                    browser.close()
                    available.append(browser_type.name)
                    logger.info(f"Detected browser: {browser_type.name}")
                except Exception as e:
                    logger.warning(f"Browser {browser_type.name} not available: {str(e)}")
                    continue
    except Exception as e:
        logger.error(f"Playwright initialization failed: {str(e)}")
        st.warning("Browser automation setup incomplete. Attempting to install browsers...")
        install_playwright_browsers()
        return detect_available_browsers()  # Try again after installation
    
    return available

def install_playwright_browsers():
    """Install Playwright browsers with proper error handling"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install"],
            capture_output=True,
            text=True,
            check=True
        )
        st.success("Browsers installed successfully")
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Browser installation failed: {e.stderr}")
        st.error(f"Browser installation failed. Please run manually:\n"
                f"`python -m playwright install`")
        return False

def get_available_browsers():
    """Detect available browsers with robust error handling"""
    try:
        with sync_playwright() as p:
            available = []
            for browser_type in [p.chromium, p.firefox, p.webkit]:
                try:
                    browser = browser_type.launch(headless=True)
                    browser.close()
                    available.append(browser_type.name)
                except Exception as e:
                    logger.warning(f"Browser {browser_type.name} not available: {e}")
            return available
    except Exception as e:
        logger.error(f"Playwright initialization failed: {e}")
        return []
        
def configure_browser_ui():
    """Enhanced browser configuration with installation fallback"""
    try:
        # First try with standard detection
        available = get_working_browser()  # Use the enhanced detection function
        
        if not available:
            st.warning("Installing required browsers...")
            subprocess.run([sys.executable, "-m", "playwright", "install"], check=True)
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
            available = get_working_browser()
        
        # Rest of your browser selection UI...
    except Exception as e:
        logger.error(f"Browser configuration failed: {str(e)}")
        return None
        
# Helper Functions
def calculate_multi_year_costs(base_cost, years, growth_rate=0.0):
    return [base_cost * (1 + growth_rate)**year for year in range(years)]

def extract_cost(value):
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        num_str = re.sub(r'[^\d.-]', '', value)
        return float(num_str) if num_str else 0.0
    return 0.0

def format_currency(value, show_inr, exchange_rate):
    if show_inr:
        return f"${value:,.2f} (₹{value*exchange_rate:,.2f})"
    return f"${value:,.2f}"

def calculate_chat_cost(sessions):
    if sessions <= 1000: return DEFAULT_CHAT_COSTS[1000] * (sessions/1000)
    elif sessions <= 5000: return DEFAULT_CHAT_COSTS[5000] * (sessions/5000)
    elif sessions <= 10000: return DEFAULT_CHAT_COSTS[10000] * (sessions/10000)
    elif sessions <= 25000: return DEFAULT_CHAT_COSTS[25000] * (sessions/25000)
    else: return DEFAULT_CHAT_COSTS[50000] * (sessions/50000)

def calculate_email_cost(emails):
    return DEFAULT_EMAIL_COST * (emails/20000) if emails <= 20000 else DEFAULT_EMAIL_COST + (emails-20000)*0.06

def get_pdf_download_link(pdf, filename):
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    b64 = base64.b64encode(pdf_bytes).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF Report</a>'

def save_plotly_figure(fig, filename):
    """Enhanced figure export with fallback"""
    try:
        # First try Kaleido
        return fig.to_image(format="png", width=1000, height=600, scale=2)
    except Exception as e:
        logger.warning(f"Kaleido failed, falling back to orca: {str(e)}")
        try:
            # Fallback to orca
            return fig.to_image(format="png", width=1000, height=600, scale=2, engine="orca")
        except:
            # Final fallback to basic matplotlib
            import matplotlib.pyplot as plt
            fig.write_image("temp.png")
            with open("temp.png", "rb") as f:
                img_bytes = f.read()
            os.remove("temp.png")
            return img_bytes
            
def generate_pdf_report(config, numeric_table, display_table, recommendation, notes, figures):
    """Generate a PDF report with all pricing details and visualizations"""
    try:
        # Convert figures to base64 strings
        cost_comparison_b64 = base64.b64encode(figures['cost_comparison']).decode()
        breakeven_b64 = base64.b64encode(figures['breakeven']).decode()
        multi_year_b64 = base64.b64encode(figures['multi_year']).decode()
        
        # Prepare table rows using string concatenation
        table_rows = []
        time_period = config['time_period']
        for _, row in display_table.iterrows():
            row_html = (
                "<tr><td>" + html.escape(str(row['Metric'])) + "</td>" +
                "<td>" + html.escape(str(row[f'Fixed_{time_period}'])) + "</td>" +
                "<td>" + html.escape(str(row[f'PAYG_{time_period}'])) + "</td></tr>"
            )
            table_rows.append(row_html)
        
        # Build HTML content piece by piece
        html_content = [
            """<!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {font-family: Arial, sans-serif; margin: 20px;}
                    h1 {color: #2c3e50;}
                    h2 {color: #34495e; margin-top: 20px;}
                    table {border-collapse: collapse; width: 100%; margin-bottom: 20px;}
                    th, td {border: 1px solid #ddd; padding: 8px; text-align: left;}
                    th {background-color: #f2f2f2;}
                    .chart {margin: 20px 0; text-align: center;}
                </style>
            </head>
            <body>
                <h1>Kcube Consulting Partners - Pricing Report</h1>
                <p>Generated on """,
                html.escape(date.today().strftime('%Y-%m-%d %H:%M:%S')),
                """</p>
                
                <h2>Configuration Parameters</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Agent Count</td><td>""",
                html.escape(str(config['agent_count'])),
                """</td></tr>
                    <tr><td>Time Period</td><td>""",
                html.escape(config['time_period']),
                """</td></tr>
                    <tr><td>Minutes per Agent</td><td>""",
                html.escape(str(config['minutes_per_agent'])),
                """ (""",
                str(round(config['minutes_per_agent']/60, 1)),
                """ hours)</td></tr>
                    <tr><td>Outbound Telephony</td><td>""",
                'Yes (+10%)' if config['outbound'] else 'No',
                """</td></tr>
                    <tr><td>Chat Sessions</td><td>""",
                html.escape(str(config['chat_sessions'])) if config['chat_sessions'] > 0 else 'Disabled',
                """</td></tr>
                    <tr><td>Email Volume</td><td>""",
                html.escape(str(config['email_volume'])) if config['email_volume'] > 0 else 'Disabled',
                """</td></tr>
                </table>
                
                <h2>Cost Comparison</h2>
                <div class="chart">
                    <img src="data:image/png;base64,""",
                cost_comparison_b64,
                """" width="800">
                </div>
                
                <h2>Cost Breakdown</h2>
                <table>
                    <tr><th>Metric</th><th>Fixed Pricing</th><th>Pay-As-You-Go</th></tr>""",
                ''.join(table_rows),
                """
                </table>
                
                <h2>Breakeven Analysis</h2>
                <div class="chart">
                    <img src="data:image/png;base64,""",
                breakeven_b64,
                """" width="800">
                </div>
                
                <h2>""",
                str(config['analysis_years']),
                """-Year Projection</h2>
                <div class="chart">
                    <img src="data:image/png;base64,""",
                multi_year_b64,
                """" width="800">
                </div>
                
                <h2>Recommendation</h2>
                <p>""",
                html.escape(recommendation).replace('\n', '<br>'),
                """</p>
                
                <h2>Notes</h2>
                <p>""",
                html.escape(notes).replace('\n', '<br>'),
                """</p>
            </body>
            </html>"""
        ]
        
        # Join all HTML parts
        html_content = ''.join(html_content)
        
        # Generate PDF using selected browser
        selected_browser = configure_browser_ui()
        if selected_browser:
            try:
                with sync_playwright() as p:
                    if selected_browser == 'firefox':
                        browser = p.firefox.launch()
                    elif selected_browser == 'webkit':
                        browser = p.webkit.launch()
                    else:
                        browser = p.chromium.launch()
                    
                    page = browser.new_page()
                    page.set_content(html_content)
                    pdf_bytes = page.pdf(
                        format='A4',
                        margin={'top': '1cm', 'bottom': '1cm'},
                        print_background=True
                    )
                    
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt="Kcube Pricing Report", ln=1, align='C')
                    return pdf
            except Exception as e:
                logger.warning(f"Browser PDF generation failed: {str(e)}")
        
        # Fall back to basic PDF generation
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Kcube Pricing Report", ln=1, align='C')
        pdf.cell(200, 10, txt=f"Generated on {date.today().strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align='C')
        
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt="Configuration Parameters", ln=1)
        pdf.cell(200, 8, txt=f"Agent Count: {config['agent_count']}", ln=1)
        pdf.cell(200, 8, txt=f"Time Period: {config['time_period']}", ln=1)
        pdf.cell(200, 8, txt=f"Minutes per Agent: {config['minutes_per_agent']} ({(config['minutes_per_agent']/60):.1f} hours)", ln=1)
        pdf.cell(200, 8, txt=f"Outbound Telephony: {'Yes (+10%)' if config['outbound'] else 'No'}", ln=1)
        pdf.cell(200, 8, txt=f"Chat Sessions: {config['chat_sessions'] if config['chat_sessions'] > 0 else 'Disabled'}", ln=1)
        pdf.cell(200, 8, txt=f"Email Volume: {config['email_volume'] if config['email_volume'] > 0 else 'Disabled'}", ln=1)
        
        pdf.ln(10)
        pdf.set_font("Arial", 'B', size=10)
        pdf.cell(200, 8, txt="Recommendation", ln=1)
        pdf.set_font("Arial", size=10)
        for line in recommendation.split('\n'):
            pdf.cell(200, 8, txt=line.strip(), ln=1)
        
        return pdf
        
    except Exception as e:
        logger.error(f"PDF generation failed: {str(e)}")
        st.error(f"Failed to generate PDF report: {str(e)}")
        return None

def get_working_browser():
    """Enhanced browser detection with Windows-specific fixes"""
    browsers_to_try = ['chromium', 'chrome', 'msedge']
    working_browsers = []
    
    for browser_name in browsers_to_try:
        try:
            with sync_playwright() as p:
                browser_type = getattr(p, browser_name)
                browser = browser_type.launch(
                    headless=False,
                    args=[
                        '--disable-gpu',
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--remote-debugging-port=9222'
                    ],
                    timeout=30000
                )
                page = browser.new_page()
                page.goto('about:blank')
                browser.close()
                working_browsers.append(browser_name)
        except Exception as e:
            logger.warning(f"Browser {browser_name} failed: {str(e)}")
    
    return working_browsers
    
def install_chrome_manually():
    """Manual Chrome installation fallback"""
    try:
        import choreographer
        from choreographer.utils import install_chrome
        logger.info("Attempting manual Chrome installation...")
        install_chrome()
        return True
    except Exception as e:
        logger.error(f"Manual Chrome installation failed: {str(e)}")
        return False

def configure_browser_with_fallbacks():
    """Final robust browser configuration"""
    # First try normal detection
    available = get_working_browser()
    
    if not available:
        st.warning("""
        No browsers detected automatically. Attempting solutions:
        1. Installing browsers...
        2. Trying manual Chrome installation...
        """)
        
        # Try automatic installation
        subprocess.run([sys.executable, "-m", "playwright", "install"], check=True)
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
        
        # Try manual Chrome installation
        if install_chrome_manually():
            available = get_working_browser()
    
    if not available:
        st.error("""
        Could not initialize any browser. Possible solutions:
        1. Run manually: `python -m playwright install`
        2. Install Chrome manually
        3. Use the basic PDF fallback
        """)
        return None
    
    # Browser selection UI
    with st.expander("⚙️ Browser Settings"):
        selected = st.selectbox(
            "Available Browsers",
            available,
            help="Choose the most stable browser for your system"
        )
        
        # Additional debug info
        if st.checkbox("Show browser debug info"):
            with sync_playwright() as p:
                browser = getattr(p, selected).launch()
                st.code(f"""
                Browser: {selected}
                Version: {browser.version}
                Executable: {browser.executable_path}
                """)
                browser.close()
    
    return selected
               
def generate_basic_pdf(config, numeric_table, display_table, recommendation, notes, figures):
    """Fallback basic PDF generation using FPDF"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Add title
        pdf.cell(200, 10, txt="Kcube Pricing Report", ln=1, align='C')
        pdf.cell(200, 10, txt=f"Generated on {date.today().strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align='C')
        
        # Add configuration
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt="Configuration Parameters", ln=1)
        pdf.cell(200, 8, txt=f"Agent Count: {config['agent_count']}", ln=1)
        pdf.cell(200, 8, txt=f"Time Period: {config['time_period']}", ln=1)
        pdf.cell(200, 8, txt=f"Minutes per Agent: {config['minutes_per_agent']} ({(config['minutes_per_agent']/60):.1f} hours)", ln=1)
        pdf.cell(200, 8, txt=f"Outbound Telephony: {'Yes (+10%)' if config['outbound'] else 'No'}", ln=1)
        pdf.cell(200, 8, txt=f"Chat Sessions: {config['chat_sessions'] if config['chat_sessions'] > 0 else 'Disabled'}", ln=1)
        pdf.cell(200, 8, txt=f"Email Volume: {config['email_volume'] if config['email_volume'] > 0 else 'Disabled'}", ln=1)
        
        # Add recommendation
        pdf.ln(10)
        pdf.set_font("Arial", 'B', size=10)
        pdf.cell(200, 8, txt="Recommendation", ln=1)
        pdf.set_font("Arial", size=10)
        for line in recommendation.split('\n'):
            pdf.cell(200, 8, txt=line.strip(), ln=1)
        
        return pdf
        
    except Exception as e:
        logger.error(f"Basic PDF generation failed: {str(e)}")
        return None

def generate_html_content(config, numeric_table, display_table, recommendation, notes, figures):
    """Generate HTML content for browser-based PDF generation"""
    # Convert figures to base64 strings
    cost_comparison_b64 = base64.b64encode(figures['cost_comparison']).decode()
    breakeven_b64 = base64.b64encode(figures['breakeven']).decode()
    multi_year_b64 = base64.b64encode(figures['multi_year']).decode()
    
    # Prepare table rows
    table_rows = []
    time_period = config['time_period']
    for _, row in display_table.iterrows():
        table_rows.append(
            f"<tr><td>{html.escape(str(row['Metric']))}</td>"
            f"<td>{html.escape(str(row[f'Fixed_{time_period}']))}</td>"
            f"<td>{html.escape(str(row[f'PAYG_{time_period}']))}</td></tr>"
        )
    
    # Process recommendation text safely
    recommendation_escaped = html.escape(recommendation)
    recommendation_with_breaks = recommendation_escaped.replace('\n', '<br>')
    
    # Define CSS as a separate raw string
    css = r"""
    body { font-family: Arial, sans-serif; margin: 20px; }
    h1 { color: #2c3e50; }
    h2 { color: #34495e; margin-top: 20px; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background-color: #f2f2f2; }
    .chart { margin: 20px 0; text-align: center; }
    """
    
    # Build HTML parts
    html_parts = [
        "<html>",
        "<head>",
        f"<style>{css}</style>",
        "</head>",
        "<body>",
        "<h1>Kcube Consulting Partners - Pricing Report</h1>",
        f"<p>Generated on {html.escape(date.today().strftime('%Y-%m-%d %H:%M:%S'))}</p>",
        "<h2>Configuration Parameters</h2>",
        "<table>",
        "<tr><th>Parameter</th><th>Value</th></tr>",
        *table_rows,
        "</table>",
        "<h2>Recommendation</h2>",
        f"<p>{recommendation_with_breaks}</p>",
        "</body>",
        "</html>"
    ]
    
    return "\n".join(html_parts)
    
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
        
        # Write recommendation
        summary_sheet.write(10, 0, 'Recommendation')
        summary_sheet.write(11, 0, recommendation.replace('\n', ' '))
        
        # Write notes
        summary_sheet.write(13, 0, 'Notes')
        summary_sheet.write(14, 0, notes.replace('\n', ' '))
    
    return output.getvalue()

def get_report_download_links(config, numeric_table, display_table, fixed_cost, payg_cost, time_period, cost_comparison_fig, breakeven_fig, multi_year_fig):
    """Generate both PDF and Excel report download links"""
    # Prepare figures
    figures = {
        'cost_comparison': save_plotly_figure(cost_comparison_fig, "cost_comparison.png"),
        'breakeven': save_plotly_figure(breakeven_fig, "breakeven.png"),
        'multi_year': save_plotly_figure(multi_year_fig, "multi_year.png")
    }
    
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
    
    # Generate notes
    notes = f"""
    Note:
    *Outbound dialing: adds 10% to the base telephony cost (Customer must provide their own dialer)
    - Chat Agent Cost: Tiered pricing (1K: $240, 5K: $240, 10K: $480, 25K: $1,200, 50K: $2,400)
    - Email Agent Cost: $1,200 for 20,000 emails ($0.06 per additional email)
    - Implementation cost is $15,000 (one-time)
    - Standard agent time: {config['minutes_per_agent']} minutes/month ({(config['minutes_per_agent']/60):.1f} hours)
    - Report generated on {date.today().strftime('%Y-%m-%d')}
    """
    
    # Generate PDF
    pdf_report = generate_pdf_report(config, numeric_table, display_table, recommendation, notes, figures)
    
    # Generate Excel
    excel_data = generate_excel_report(config, numeric_table, display_table, recommendation, notes)
    
    # Create download links
    if pdf_report:
        pdf_link = get_pdf_download_link(pdf_report, f"kcube_report_{config['agent_count']}_agents.pdf")
    else:
        pdf_link = "<p style='color:red'>PDF generation failed - check browser configuration</p>"
    
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
    fixed_monthly_cost = float(fixed['MonthlyCost'].values[0])
    payg_monthly_cost = float(payg['MonthlyCost'].values[0])
    
    # Create numeric dataframe for calculations
    numeric_data = {
        'Metric': metrics,
        'Fixed_Monthly_Value': [
            fixed['Agents'].values[0],
            minutes_per_agent * fixed['Agents'].values[0],
            round(minutes_per_agent * fixed['Agents'].values[0] / 60, 1),
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
            payg['Agents'].values[0],
            minutes_per_agent * payg['Agents'].values[0],
            round(minutes_per_agent * payg['Agents'].values[0] / 60, 1),
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
            str(fixed['Agents'].values[0]),
            f"{minutes_per_agent * fixed['Agents'].values[0]:,.0f}",
            f"{round(minutes_per_agent * fixed['Agents'].values[0] / 60, 1):,.1f}",
            f"${fixed_monthly_cost * 0.9:,.2f}{'*' if outbound_toggle else ''}",
            "Included",
            "Included",
            "Included",
            "$15,000 (One-time)",
            f"${calculate_chat_cost(chat_sessions):,.2f}" if chat_sessions > 0 else "Not enabled",
            f"${calculate_email_cost(email_volume):,.2f}" if email_volume > 0 else "Not enabled",
            f"${fixed_monthly_cost:,.2f}"
        ],
        'PAYG_Monthly': [
            str(payg['Agents'].values[0]),
            f"{minutes_per_agent * payg['Agents'].values[0]:,.0f}",
            f"{round(minutes_per_agent * payg['Agents'].values[0] / 60, 1):,.1f}",
            f"${payg_monthly_cost * 0.9:,.2f}{'*' if outbound_toggle else ''}",
            "Included",
            "Included",
            "Included",
            "$15,000 (One-time)",
            f"${calculate_chat_cost(chat_sessions):,.2f}" if chat_sessions > 0 else "Not enabled",
            f"${calculate_email_cost(email_volume):,.2f}" if email_volume > 0 else "Not enabled",
            f"${payg_monthly_cost:,.2f}"
        ],
        'Fixed_Yearly': [
            str(fixed['Agents'].values[0]),
            f"{minutes_per_agent * fixed['Agents'].values[0] * 12:,.0f}",
            f"{round(minutes_per_agent * fixed['Agents'].values[0] * 12 / 60, 1):,.1f}",
            f"${fixed_monthly_cost * 0.9 * 12:,.2f}{'*' if outbound_toggle else ''}",
            "Included",
            "Included",
            "Included",
            "$15,000 (One-time)",
            f"${calculate_chat_cost(chat_sessions) * 12:,.2f}" if chat_sessions > 0 else "Not enabled",
            f"${calculate_email_cost(email_volume) * 12:,.2f}" if email_volume > 0 else "Not enabled",
            f"${fixed_monthly_cost * 12:,.2f}"
        ],
        'PAYG_Yearly': [
            str(payg['Agents'].values[0]),
            f"{minutes_per_agent * payg['Agents'].values[0] * 12:,.0f}",
            f"{round(minutes_per_agent * payg['Agents'].values[0] * 12 / 60, 1):,.1f}",
            f"${payg_monthly_cost * 0.9 * 12:,.2f}{'*' if outbound_toggle else ''}",
            "Included",
            "Included",
            "Included",
            "$15,000 (One-time)",
            f"${calculate_chat_cost(chat_sessions) * 12:,.2f}" if chat_sessions > 0 else "Not enabled",
            f"${calculate_email_cost(email_volume) * 12:,.2f}" if email_volume > 0 else "Not enabled",
            f"${payg_monthly_cost * 12:,.2f}"
        ]
    }
    
    return pd.DataFrame(numeric_data), pd.DataFrame(display_data)

# Main Dashboard
def main():
    # Configure page
    st.set_page_config(page_title="Kcube Pricing Dashboard", page_icon="📊", layout="wide")
    st.title("Kcube Consulting Partners - Pricing Options (Fixed & PayG)")
    
    # Initialize browser configuration
    configure_browser_ui()
    
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
        usd_to_inr = st.number_input("USD to INR Exchange Rate", min_value=1.0, value=87.0, step=0.5)
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

    processed_df = process_pricing_data(df, chat_sessions, email_volume, outbound_toggle, minutes_per_agent)
    numeric_table, display_table = create_detailed_cost_table(
        processed_df, selected_agent, chat_sessions, email_volume, 
        outbound_toggle, minutes_per_agent
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Cost Comparison", "📝 Detailed Breakdown", "⚖️ Breakeven Analysis", "📈 Multi-Year Projection", "📁 Raw Data"])

    with tab1:
        st.markdown(f"### 💵 {time_period} Cost Comparison{'*' if outbound_toggle else ''}")
        
        fixed_cost = processed_df[(processed_df['Agents'] == selected_agent) & (processed_df['Option'] == 'Fixed Pricing')][f"{time_period}Cost"].values[0]
        payg_cost = processed_df[(processed_df['Agents'] == selected_agent) & (processed_df['Option'] == 'Pay-As-You-Go')][f"{time_period}Cost"].values[0]
        
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
        config = {
            'agent_count': selected_agent,
            'time_period': time_period,
            'outbound': outbound_toggle,
            'chat_sessions': chat_sessions,
            'email_volume': email_volume,
            'minutes_per_agent': minutes_per_agent,
            'analysis_years': analysis_years,
            'growth_rate': growth_rate
        }
        
        compare_df = processed_df.pivot_table(
            index='Agents',
            columns='Option',
            values=f"{time_period}Cost",
            aggfunc='sum'
        ).reset_index()
        compare_df['Savings'] = compare_df['Fixed Pricing'] - compare_df['Pay-As-You-Go']
        compare_df['Breakeven Months'] = (IMPLEMENTATION_COST / compare_df['Savings']).round(1)
        
        breakeven_fig = px.line(
            compare_df,
            x='Agents',
            y='Breakeven Months',
            title=f"Months to Recover Implementation Cost{'*' if outbound_toggle else ''}",
            markers=True
        )
        breakeven_fig.add_hline(y=12, line_dash="dash", line_color="red", annotation_text="1 Year Benchmark")
        breakeven_fig.add_vline(x=selected_agent-0.5, line_width=3, line_dash="dash", line_color="red")
        
        fixed_yearly = processed_df[(processed_df['Agents'] == selected_agent) & (processed_df['Option'] == 'Fixed Pricing')]['YearlyCost'].values[0]
        payg_yearly = processed_df[(processed_df['Agents'] == selected_agent) & (processed_df['Option'] == 'Pay-As-You-Go')]['YearlyCost'].values[0]
        
        years = list(range(1, analysis_years + 1))
        fixed_costs = calculate_multi_year_costs(fixed_yearly, analysis_years, growth_rate)
        payg_costs = calculate_multi_year_costs(payg_yearly, analysis_years, growth_rate)
        
        multi_year_fig = make_subplots(specs=[[{"secondary_y": True}]], subplot_titles=("Cumulative Costs", "Annual Costs"))
        multi_year_fig.add_trace(
            go.Scatter(
                x=years,
                y=np.cumsum(fixed_costs),
                name="Fixed Pricing (Cumulative)",
                line=dict(color='blue', width=4),
                hovertemplate="Year %{x}<br>Total: $%{y:,.2f}"
            ),
            secondary_y=False
        )
        multi_year_fig.add_trace(
            go.Scatter(
                x=years,
                y=np.cumsum(payg_costs),
                name="PayG (Cumulative)",
                line=dict(color='green', width=4),
                hovertemplate="Year %{x}<br>Total: $%{y:,.2f}"
            ),
            secondary_y=False
        )
        multi_year_fig.add_trace(
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
        multi_year_fig.add_trace(
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
        multi_year_fig.add_hline(
            y=IMPLEMENTATION_COST,
            line_dash="dot",
            line_color="red",
            annotation_text="Implementation Cost",
            annotation_position="bottom right",
            secondary_y=False
        )
        
        pdf_link, excel_link = get_report_download_links(
            config, numeric_table, display_table, 
            fixed_cost, payg_cost, time_period,
            cost_comparison_fig, breakeven_fig, multi_year_fig
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(pdf_link, unsafe_allow_html=True)
        with col2:
            st.markdown(excel_link, unsafe_allow_html=True)

    with tab3:
        st.markdown(f"### ⚖️ Breakeven Analysis{'*' if outbound_toggle else ''}")
        
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

    with tab4:
        st.markdown(f"### 📈 {analysis_years}-Year Cost Projection{'*' if outbound_toggle else ''}")

        fixed_yearly = processed_df[(processed_df['Agents'] == selected_agent) & (processed_df['Option'] == 'Fixed Pricing')]['YearlyCost'].values[0]
        payg_yearly = processed_df[(processed_df['Agents'] == selected_agent) & (processed_df['Option'] == 'Pay-As-You-Go')]['YearlyCost'].values[0]

        years = list(range(1, analysis_years + 1))
        fixed_costs = calculate_multi_year_costs(fixed_yearly, analysis_years, growth_rate)
        payg_costs = calculate_multi_year_costs(payg_yearly, analysis_years, growth_rate)

        fig = make_subplots(specs=[[{"secondary_y": True}]], subplot_titles=("Cumulative Costs", "Annual Costs"))

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
    fixed_cost = processed_df[(processed_df['Agents'] == selected_agent) & (processed_df['Option'] == 'Fixed Pricing')][f"{time_period}Cost"].values[0]
    payg_cost = processed_df[(processed_df['Agents'] == selected_agent) & (processed_df['Option'] == 'Pay-As-You-Go')][f"{time_period}Cost"].values[0]
    
    if payg_cost < fixed_cost:
        savings = fixed_cost - payg_cost
        breakeven = (IMPLEMENTATION_COST / savings) if savings > 0 else 0
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
    - Chat Agent Cost: Tiered pricing (1K: $240, 5K: $240, 10K: $480, 25K: $1,200, 50K: $2,400)  
    - Email Agent Cost: $1,200 for 20,000 emails ($0.06 per additional email)  
    - Implementation cost is $15,000 (one-time)  
    - Standard agent time: {minutes_per_agent} minutes/month ({(minutes_per_agent/60):.1f} hours)  
    - Report generated on {date.today().strftime('%Y-%m-%d')}  
    """)

if __name__ == "__main__":
    main()