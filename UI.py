import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import json
from fpdf import FPDF
import io
import plotly.graph_objects as go
import plotly.figure_factory as ff
from PIL import Image
import numpy as np
import os 
import tempfile
import plotly.io as pio
from dotenv import load_dotenv
load_dotenv()
# Assuming you have set up Groq API access
from groq import Groq
api_key = os.getenv('GROQ_API_KEY')
# Set up Groq client (you'll need to handle API key securely)
groq_client = Groq(api_key=api_key)

def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None
    return df

def preprocess_data(df):
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Handle missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    # Convert date columns to datetime
    date_columns = df.select_dtypes(include=['object']).columns
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            pass
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    return df


def validate_data(df):
    def serialize(obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    validation_results = {
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),  # Convert to int
        "column_types": df.dtypes.astype(str).to_dict(),
        "unique_values": {col: int(df[col].nunique()) for col in df.columns},  # Convert to int
        "value_ranges": {
            col: {"min": serialize(df[col].min()), "max": serialize(df[col].max())}
            for col in df.select_dtypes(include=[np.number]).columns
        }
    }
    return validation_results

def generate_llm_report(df, prompt, validation_results):
    # Convert DataFrame to JSON
    df_json = df.to_json(orient='records', date_format='iso')
    
    # Prepare the message for the LLM
    message = f"""
    Analyze the following billing data and validation results:
    
    Data: {df_json}
    
    Validation Results: {json.dumps(validation_results, default=str)}
    
    User Prompt: {prompt}
    
    Generate a detailed report analyzing the billing data, addressing the user's prompt, 
    and providing insights based on the data and validation results. 
    
    Structure your report with the following sections, using double asterisks (**) to denote section headers:
    
    1. **Summary**
    2. **Spending Trends**
    3. **Insights**
    4. **Recommendations**
    """
    # Call Groq API
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a data analyst specializing in billing data analysis."},
            {"role": "user", "content": message}
        ],
        model="llama3-70b-8192",
    )
    
    # After getting the response from the LLM
    report_text = response.choices[0].message.content
    
    # Create visualizations
    figures = create_visualizations(df)
    
    # Combine report text with visualizations
    combined_report = report_text
    # for title, _ in figures:
    #     combined_report += f"\n\n[Insert {title} here]\n\n"
    
    return combined_report, figures


class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Billing Data Analysis Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf_report(report_text, figures):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Split the report text into sections
    sections = report_text.split('\n\n')
    
    for i, section in enumerate(sections):
        # Check if the section is a header
        if section.strip().startswith('**') and section.strip().endswith('**'):
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, section.strip('*'), 0, 1)
            pdf.set_font("Arial", size=12)
        else:
            pdf.multi_cell(0, 10, section)
        
        # After each section, check if we have a figure to insert
        if i < len(figures):
            fig_title, fig = figures[i]
            img_bytes = fig.to_image(format="png", scale=2)  # Increased scale for better quality
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                tmpfile.write(img_bytes)
                tmpfile_name = tmpfile.name

            # Add a new page for the figure if near the bottom of the page
            if pdf.get_y() > 180:
                pdf.add_page()

            # Add the image to the PDF
            pdf.image(tmpfile_name, x=10, y=pdf.get_y()+10, w=190)
            pdf.set_y(pdf.get_y() + 140)  # Increased space after image
            pdf.set_font("Arial", 'I', 10)
            pdf.cell(0, 10, fig_title, 0, 1, 'C')  # Add figure title
            pdf.set_font("Arial", size=12)
            pdf.ln(10)  # Add extra space after the figure
            
            # Remove the temporary file
            os.unlink(tmpfile_name)
    
    pdf_output = pdf.output(dest='S').encode('latin-1')
    return pdf_output

def create_visualizations(df):
    figures = []

    # Set a white background for all plots
    pio.templates.default = "plotly_white"

    # Helper function to find a column by potential names
    def find_column(potential_names):
        for name in potential_names:
            if name in df.columns:
                return name
        return None

    # 1. Histogram of Billing Amounts
    amount_col = find_column(['Amount', 'amount', 'billing_amount', 'total'])
    if amount_col:
        fig = px.histogram(df, x=amount_col, title=f'Distribution of {amount_col}')
        fig.update_traces(marker_line_color="black", marker_line_width=1)
        figures.append((f"Histogram of {amount_col}", fig))

    # 2. Bar Chart of Department-wise Billing
    dept_col = find_column(['Department', 'department', 'dept'])
    if dept_col and amount_col:
        dept_billing = df.groupby(dept_col)[amount_col].sum().sort_values(ascending=False)
        fig = px.bar(dept_billing, x=dept_billing.index, y=dept_billing.values, 
                     title=f'Total {amount_col} by {dept_col}')
        figures.append((f"Bar Chart of {dept_col}-wise {amount_col}", fig))

    # 3. Heatmap of Customer Billing Activity
    customer_col = find_column(['Customer Name', 'customer', 'client'])
    date_col = find_column(['Billing Date', 'date', 'transaction_date'])
    if customer_col and date_col and amount_col:
        customer_activity = df.pivot_table(values=amount_col, index=customer_col, 
                                           columns=date_col, aggfunc='sum', fill_value=0)
        fig = px.imshow(customer_activity, title=f'Heatmap of {customer_col} {amount_col} Activity')
        figures.append((f"Heatmap of {customer_col} {amount_col} Activity", fig))

    # 4. Timeline of Billing Cycle
    if date_col and amount_col:
        df[date_col] = pd.to_datetime(df[date_col])
        daily_billing = df.groupby(date_col)[amount_col].sum().reset_index()
        fig = px.line(daily_billing, x=date_col, y=amount_col, title=f'Timeline of {amount_col} Cycle')
        figures.append((f"Timeline of {amount_col} Cycle", fig))

    # 5. Pie Chart of Payment Status
    status_col = find_column(['Status', 'status', 'payment_status'])
    if status_col:
        status_counts = df[status_col].value_counts()
        fig = px.pie(values=status_counts.values, names=status_counts.index, 
                     title=f'Distribution of {status_col}')
        figures.append((f"Pie Chart of {status_col}", fig))

    # Ensure all figures have a white background
    for _, fig in figures:
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')

    return figures


def main():
    st.title("Billing Data Analysis Report Generator")

    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = None

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.write("Data Preview (Before Preprocessing):")
            st.write(df.head())

            # Initialize validation_results here
            st.session_state.validation_results = validate_data(df)

            st.subheader("Data Preprocessing")
            if st.button("Preprocess Data"):
                st.session_state.df = preprocess_data(df)
                st.session_state.validation_results = validate_data(st.session_state.df)
                st.write("Data Preview (After Preprocessing):")
                st.write(st.session_state.df.head())
                st.write("Validation Results:")
                st.json(st.session_state.validation_results)

    st.subheader("Analysis Prompt")
    prompt = st.text_area("Enter your question or what you'd like to know about the billing data:", 
    """Analyze the billing data to ensure accuracy and timeliness.Identify any discrepancies or unusual patterns You can assume and utilize relevant metrics to analyze the billing data, such as billing cycle times, payment discrepancies, late  payment occurrences, and average payment amounts.""", height=100)
    
    if st.button("Generate Report"):
        if prompt and st.session_state.df is not None:
            with st.spinner("Generating report..."):
                report_text, figures = generate_llm_report(st.session_state.df, prompt, st.session_state.validation_results)
                
                st.subheader("Generated Report")
                st.write(report_text)
                
                for title, fig in figures:
                    st.subheader(title)
                    st.plotly_chart(fig)
                
                pdf_output = create_pdf_report(report_text, figures)
                
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_output,
                    file_name="billing_analysis_report.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("Please enter an analysis prompt and ensure data is preprocessed before generating the report.")

if __name__ == "__main__":
    main()
