import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from dotenv import load_dotenv
from pandasai_litellm.litellm import LiteLLM
import pandasai as pai
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from fpdf import FPDF
import tempfile
import datetime

# Load environment variables
load_dotenv()

def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty. Please upload a valid dataset.")
    except ValueError:
        st.error("Invalid file format or corrupted data.")
    except Exception as e:
        st.error(f"Error loading file: {e}")
    return None

def main():
    st.set_page_config(page_title="VizAI Analytics", layout="wide")
    st.title("📊 VizAI - Advanced Business Data Analytics Tool")

    with st.sidebar:
        st.header("1. Upload Data")
        uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
        st.markdown("---")
        
        # --- PHASE 2: Export Module (Placeholder setup) ---
        st.header("2. Export & Report")
        export_container = st.container()

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            if df.empty or len(df.columns) == 0:
                st.error("The uploaded file is empty. Please add data before uploading.")
            else:
                # --- Capture Pre-Cleaning Stats ---
                pre_clean_rows = len(df)
                pre_clean_missing = df.isnull().sum()

                # Data Cleaning
                df = df.dropna(how='all')
                numeric_cols = df.select_dtypes(include=['number']).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                categorical_cols = df.select_dtypes(exclude=['number']).columns
                df[categorical_cols] = df[categorical_cols].fillna('Unknown')
                st.sidebar.success("File uploaded successfully!")

                # --- PHASE 2: Export Implementations ---
                with export_container:
                    # 1. CSV Export
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Cleaned CSV", data=csv, file_name="vizai_cleaned_data.csv", mime="text/csv")
                    
                    # 2. Comprehensive PDF Report Generation
                    if st.button("Generate Executive PDF Report"):
                        with st.spinner("Compiling comprehensive report..."):
                            try:
                                pdf = FPDF()
                                pdf.add_page()
                                
                                # --- TITLE & METADATA ---
                                pdf.set_font("Arial", 'B', 16)
                                pdf.cell(0, 10, "VizAI Executive Data Report", ln=1, align='C')
                                pdf.set_font("Arial", 'I', 10)
                                pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align='C')
                                pdf.ln(5)
                                
                                # --- SECTION 1: PRE-CLEANING OVERVIEW ---
                                pdf.set_font("Arial", 'B', 12)
                                pdf.cell(0, 10, "1. Pre-Cleaning Data Overview", ln=1)
                                pdf.set_font("Arial", '', 11)
                                pdf.cell(0, 8, f"Initial Records (Rows): {pre_clean_rows}", ln=1)
                                pdf.cell(0, 8, f"Total Features (Columns): {len(df.columns)}", ln=1)
                                
                                cols_with_missing_pre = pre_clean_missing[pre_clean_missing > 0]
                                if cols_with_missing_pre.empty:
                                    pdf.cell(0, 8, "Initial Data Quality: Excellent (No missing values).", ln=1)
                                else:
                                    pdf.cell(0, 8, "Missing values detected before cleaning:", ln=1)
                                    for col, count in cols_with_missing_pre.items():
                                        pdf.cell(0, 8, f"  - {col}: {count} missing", ln=1)
                                pdf.ln(5)
                                
                                # --- SECTION 2: POST-CLEANING OVERVIEW ---
                                pdf.set_font("Arial", 'B', 12)
                                pdf.cell(0, 10, "2. Post-Cleaning Data Overview", ln=1)
                                pdf.set_font("Arial", '', 11)
                                pdf.cell(0, 8, f"Final Records (Rows): {len(df)}", ln=1)
                                
                                missing_info = df.isnull().sum()
                                cols_with_missing = missing_info[missing_info > 0]
                                if cols_with_missing.empty:
                                    pdf.cell(0, 8, "Data Quality: Excellent (No missing values detected post-cleaning).", ln=1)
                                else:
                                    pdf.cell(0, 8, "Remaining missing values:", ln=1)
                                    for col, count in cols_with_missing.items():
                                        pdf.cell(0, 8, f"  - {col}: {count} missing", ln=1)
                                pdf.ln(5)
                                
                                # --- SECTION 3: KEY STATISTICS ---
                                pdf.set_font("Arial", 'B', 12)
                                pdf.cell(0, 10, "3. Key Statistical Metrics (Top Numeric Fields)", ln=1)
                                pdf.set_font("Arial", '', 11)
                                for col in numeric_cols[:5]: # Limit to top 5
                                    mean_val = df[col].mean()
                                    max_val = df[col].max()
                                    min_val = df[col].min()
                                    pdf.cell(0, 8, f"{col.capitalize()}: Mean = {mean_val:.2f} | Min = {min_val:.2f} | Max = {max_val:.2f}", ln=1)
                                pdf.ln(5)
                                
                                # --- SECTION 4: VISUALIZATIONS ---
                                pdf.set_font("Arial", 'B', 12)
                                pdf.cell(0, 10, "4. AI-Generated Visualizations", ln=1)
                                pdf.set_font("Arial", '', 11)
                                
                                chart_dir = "exports/charts"
                                if os.path.exists(chart_dir):
                                    charts = [f for f in os.listdir(chart_dir) if f.endswith('.png')]
                                    if not charts:
                                        pdf.cell(0, 8, "No charts were generated by the AI during this session.", ln=1)
                                    else:
                                        pdf.cell(0, 8, f"Successfully captured {len(charts)} chart(s).", ln=1)
                                        for chart in charts:
                                            pdf.add_page()
                                            pdf.set_font("Arial", 'B', 12)
                                            pdf.cell(0, 10, f"Exhibit: {chart}", ln=1)
                                            pdf.image(os.path.join(chart_dir, chart), x=10, y=30, w=190)
                                
                                # --- OUTPUT PREPARATION ---
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                    pdf.output(tmp_file.name)
                                    with open(tmp_file.name, "rb") as f:
                                        pdf_data = f.read()
                                        
                                st.download_button("Download Executive PDF", data=pdf_data, file_name="vizai_executive_report.pdf", mime="application/pdf")
                            except Exception as e:
                                st.error(f"Error generating PDF: {e}")

                # Ensure charts folder exists
                os.makedirs("exports/charts", exist_ok=True)
                
                # AI Setup
                api_key = os.getenv("GEMINI_API_KEY")
                sdf = None
                if not api_key:
                    st.warning("⚠️ GEMINI_API_KEY is missing. Please add it to your .env file.")
                else:
                    try:
                        llm = LiteLLM(model="gemini/gemini-2.5-flash", api_key=api_key)
                        pai.config.set({"llm": llm, "save_charts": True, "save_charts_path": "exports/charts"})
                        sdf = pai.SmartDataframe(df)
                    except Exception as e:
                        st.error(f"Failed to initialize AI: {e}")

                # Create Tabs (Added Tab 4 for Advanced Analytics)
                tab1, tab2, tab3, tab4 = st.tabs(["Data Profiling", "Manual Visualization", "AI Chat", "Advanced Analytics 🚀"])

                # --- TAB 1: Data Profiling ---
                with tab1:
                    st.subheader("Raw Data Preview")
                    st.dataframe(df.head(10))
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Statistical Summary")
                        st.write(df.describe())
                    with col2:
                        st.subheader("Data Types & Missing Values")
                        info_df = pd.DataFrame({'Data Type': df.dtypes.astype(str), 'Missing Values': df.isnull().sum()})
                        st.dataframe(info_df)

                # --- TAB 2: Manual Visualization ---
                with tab2:
                    st.subheader("Create Custom Charts")
                    columns = df.columns.tolist()
                    col_x, col_y, col_type = st.columns(3)
                    with col_x:
                        x_axis = st.selectbox("Select X-axis", options=["Select..."] + columns)
                    with col_y:
                        y_axis = st.selectbox("Select Y-axis", options=["Select..."] + columns)
                    with col_type:
                        chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Line Chart", "Scatter Plot"])
                    
                    if x_axis != "Select..." and y_axis != "Select...":
                        try:
                            if chart_type == "Bar Chart":
                                fig = px.bar(df, x=x_axis, y=y_axis)
                            elif chart_type == "Line Chart":
                                fig = px.line(df, x=x_axis, y=y_axis)
                            elif chart_type == "Scatter Plot":
                                fig = px.scatter(df, x=x_axis, y=y_axis)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not generate chart: {e}")

                # --- TAB 3: AI Chat Interface ---
                with tab3:
                    st.subheader("💬 Ask your Data")
                    if sdf is None:
                        st.info("AI functionality is disabled. Please check your API key.")
                    else:
                        if "messages" not in st.session_state:
                            st.session_state.messages = []
                        for message in st.session_state.messages:
                            with st.chat_message(message["role"]):
                                if message.get("is_image"):
                                    st.image(message["content"])
                                else:
                                    st.markdown(message["content"])

                        if prompt := st.chat_input("Ask a question about your data..."):
                            st.session_state.messages.append({"role": "user", "content": prompt, "is_image": False})
                            with st.chat_message("user"):
                                st.markdown(prompt)

                            with st.chat_message("assistant"):
                                with st.spinner("Analyzing data..."):
                                    try:
                                        response = sdf.chat(prompt)
                                        if isinstance(response, str) and response.endswith('.png'):
                                            st.image(response)
                                            st.session_state.messages.append({"role": "assistant", "content": response, "is_image": True})
                                        else:
                                            st.write(response)
                                            st.session_state.messages.append({"role": "assistant", "content": str(response), "is_image": False})
                                    except Exception as e:
                                        st.error("The AI couldn't process that request.")
                                        st.error(str(e))

                # --- TAB 4: Advanced Analytics (PHASE 2) ---
                with tab4:
                    st.subheader("🔍 Anomaly Detection")
                    st.write("Automatically flag unusual spikes or drops in your numeric data.")
                    if st.button("Run Anomaly Detection"):
                        if len(numeric_cols) > 0:
                            with st.spinner("Scanning for anomalies..."):
                                iso_model = IsolationForest(contamination=0.05, random_state=42)
                                df['Anomaly_Score'] = iso_model.fit_predict(df[numeric_cols])
                                anomalies = df[df['Anomaly_Score'] == -1]
                                
                                st.error(f"⚠️ Detected {len(anomalies)} anomalies in the dataset.")
                                st.dataframe(anomalies.drop(columns=['Anomaly_Score']))
                                
                                # Plot anomalies if we have at least 2 numeric columns
                                if len(numeric_cols) >= 2:
                                    fig_anom = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                                                          color=df['Anomaly_Score'].astype(str),
                                                          color_discrete_map={'-1': 'red', '1': 'blue'},
                                                          title="Anomaly Distribution (Red = Anomaly)")
                                    st.plotly_chart(fig_anom, use_container_width=True)
                        else:
                            st.warning("No numeric columns available for anomaly detection.")

                    st.markdown("---")
                    
                    st.subheader("📈 Predictive Forecasting")
                    st.write("Forecast future trends using Random Forest Machine Learning.")
                    
                    col_dt, col_tg, col_hz = st.columns(3)
                    with col_dt:
                        date_col = st.selectbox("Select Date Column", ["Select..."] + df.columns.tolist())
                    with col_tg:
                        target_col = st.selectbox("Select Target Metric", ["Select..."] + numeric_cols.tolist())
                    with col_hz:
                        horizon = st.slider("Forecast Horizon (Days)", 7, 90, 30)

                    if st.button("Generate Forecast"):
                        if date_col != "Select..." and target_col != "Select...":
                            try:
                                with st.spinner("Training ML Model..."):
                                    # Prepare Data
                                    ml_df = df.copy()
                                    ml_df[date_col] = pd.to_datetime(ml_df[date_col])
                                    ml_df = ml_df.sort_values(by=date_col)
                                    ml_df['ordinal'] = ml_df[date_col].apply(lambda x: x.toordinal())
                                    
                                    X = ml_df[['ordinal']]
                                    y = ml_df[target_col]
                                    
                                    # Train Model
                                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                                    model.fit(X, y)
                                    
                                    # Predict
                                    last_date = ml_df[date_col].max()
                                    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)]
                                    future_X = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
                                    predictions = model.predict(future_X)
                                    
                                    # Plotting Data Setup
                                    ml_df['Type'] = 'Historical'
                                    future_df = pd.DataFrame({date_col: future_dates, target_col: predictions, 'Type': 'Forecast'})
                                    combined_df = pd.concat([ml_df[[date_col, target_col, 'Type']], future_df])
                                    
                                    fig_pred = px.line(combined_df, x=date_col, y=target_col, color='Type', 
                                                       line_dash='Type', title=f"{target_col} Forecast for next {horizon} days")
                                    st.plotly_chart(fig_pred, use_container_width=True)
                            except Exception as e:
                                st.error("Error generating forecast. Ensure the Date column contains valid dates.")
                        else:
                            st.warning("Please select both a Date column and a Target Metric.")
    else:
        st.info("👋 Welcome! Please upload a .csv or .xlsx file on the sidebar to begin analyzing your data.")

if __name__ == "__main__":
    main()