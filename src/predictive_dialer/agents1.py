import os
import asyncio
import re
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from agents import set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv
import pandas as pd
import openpyxl
import streamlit as st
import json

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

@function_tool
def excel_data():
    """Read and return all Excel data (with column names and full rows)"""
    st.write("excel_data tool called")  # Debugging log
    try:
        if 'uploaded_file' not in st.session_state or st.session_state.uploaded_file is None:
            return "No file uploaded"
        
        df = pd.read_excel(st.session_state.uploaded_file, engine='openpyxl')
        df.columns = [col.strip().lower() for col in df.columns]
        st.write("Excel data loaded successfully:", df.columns.tolist())
        
        return {
            "columns": df.columns.tolist(),
            "data": df.to_dict(orient='records'),
            "sample": df.head().to_dict(orient='records')
        }
    except Exception as e:
        st.write(f"Error in excel_data: {str(e)}")
        return f"Error reading Excel file: {str(e)}"

async def manual_simulate_call(name: str, phone: str) -> str:
    """Manually simulate a phone call to a lead"""
    st.write("manual_simulate_call called for", name, phone)  # Debugging log
    if not phone or not isinstance(phone, str) or not re.match(r'^\d{10,}$', phone):
        return f"Skipping {name}: No valid phone number"
    st.write(f"Calling {name} at {phone}...")
    await asyncio.sleep(2)
    st.write(f"Call completed for {name}")
    return f"Call completed for {name}"

external_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

set_tracing_disabled(disabled=True)

async def main():
    response_container = st.empty()
    
    lead_prioritizing_agent = Agent(
        name="Lead Prioritizing Agent",
        instructions='''
        You are an expert in analyzing and prioritizing leads from Excel data. Follow these steps:
        1. You have received a column mapping in this format: {"name_column": "leads", "phone_column": "phone number", "status_column": "status"}.
        2. Use the excel_data tool to load the Excel data.
        3. Using the column mapping:
           - Find lead names in the "leads" column.
           - Find lead status in the "status" column (values like "Qualified", "New", "Pending").
           - Find sales records in the "sales record of client" column.
        4. Sort leads by status ("Qualified" first, then "New", then "Pending") and within each status, sort by sales record (highest first).
        5. Return the results in this exact JSON format:
        {
            "prioritized_leads": [
                {
                    "name": "Lead Name",
                    "status": "Status",
                    "sales_record": "Value",
                    "priority": 1
                }
            ]
        }
        If you cannot find the required columns or data, use the excel_data tool again and explain the issue.
        ''',
        tools=[excel_data],
        handoff_description="Expert in analyzing and prioritizing leads from Excel data.",
        model=model
    )

    calling_agent = Agent(
        name="Calling Agent",
        instructions='''
        You receive a prioritized list of leads in JSON format. For each lead:
        1. Use excel_data to find the phone number based on the lead's name.
        2. Simulate the call by printing a message (e.g., "Calling {name} at {phone}...").
        3. Return a summary in this format:
        {
            "call_results": [
                {
                    "name": "Lead Name",
                    "phone": "Phone Number",
                    "status": "Call Status"
                }
            ]
        }
        If no phone number is found, skip the lead and note it in the status.
        ''',
        tools=[excel_data],
        handoff_description="Expert in making and tracking calls to leads.",
        model=model
    )

    excel_column_agent = Agent(
        name="Excel Column Detector",
        instructions='''
        Analyze the Excel file structure:
        1. Use the excel_data tool to load the file.
        2. Identify columns for:
           - Lead names (e.g., 'name', 'leads')
           - Phone numbers (e.g., 'phone number' with 10+ digit data)
           - Lead status (e.g., 'status' with 'Qualified', 'New', 'Pending')
        3. Return the mapping in this format:
        {
            "name_column": "column_name",
            "phone_column": "column_name",
            "status_column": "column_name"
        }
        If a column is not found, set it to null.
        ''',
        tools=[excel_data],
        model=model
    )

    manager_agent = Agent(
        name="Manager",
        instructions='''
        Coordinate the lead prioritization and calling process:
        1. Use Excel Column Detector to identify the correct columns.
        2. Pass the column mapping to Lead Prioritizing Agent.
        3. Pass the prioritized list to Calling Agent.
        4. Ensure all steps complete successfully.
        If any step fails, use the excel_data tool to debug and explain the issue.
        ''',
        handoffs=[excel_column_agent, lead_prioritizing_agent, calling_agent],
        model=model
    )

    result = Runner.run_streamed(manager_agent, input='Please analyze the Excel file, prioritize the leads, and simulate calls')
    prioritized_text = ""
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            prioritized_text += event.data.delta
            response_container.write(prioritized_text)

    # Do not display the raw response to the user
    # st.write("Raw response:", prioritized_text)  # Removed for non-technical users
    
    try:
        json_match = re.search(r'\{.*\}', prioritized_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if 'prioritized_leads' in data:
                lead_names = [lead['name'] for lead in data['prioritized_leads']]
                st.success(f"Found {len(lead_names)} leads to process")
                # Process calls
                df = pd.read_excel(st.session_state.uploaded_file, engine='openpyxl')
                df.columns = [col.strip().lower() for col in df.columns]
                output_df = df.copy()
                output_df['Call Status'] = None
                
                st.write("### Lead Prioritization and Call Results:")
                for lead in data['prioritized_leads']:
                    name = lead['name']
                    status = lead['status']
                    sales_record = lead['sales_record']
                    priority = lead['priority']
                    st.write(f"Priority {priority}: {name} (Status: {status}, Sales: {sales_record})")
                    
                    lead_row = df[df['leads'] == name]
                    if not lead_row.empty:
                        phone = lead_row.iloc[0]['phone number']
                        call_result = await manual_simulate_call(name, phone)
                        output_df.loc[output_df['leads'] == name, 'Call Status'] = call_result
                        st.write(f"- {call_result}")
                    else:
                        st.warning(f"Lead {name} not found in Excel file.")
                
                st.write("### Updated Lead Data:")
                st.dataframe(output_df)
                output_file = "updated_leads.xlsx"
                output_df.to_excel(output_file, index=False)
                st.success(f"Excel file updated with call status. Saved as {output_file}")
            else:
                st.error("No prioritized leads found in the response.")
                st.write("Falling back to manual prioritization...")
                df = pd.read_excel(st.session_state.uploaded_file, engine='openpyxl')
                df.columns = [col.strip().lower() for col in df.columns]
                
                # Manual prioritization
                status_order = {'qualified': 0, 'new': 1, 'pending': 2}
                df['status_order'] = df['status'].str.lower().map(status_order).fillna(3)
                prioritized = df.sort_values(by=['status_order', 'sales record of client'], 
                                           ascending=[True, False])
                prioritized = prioritized.drop(columns=['status_order'])
                
                # Generate prioritized leads list
                prioritized_leads = []
                for idx, row in prioritized.iterrows():
                    prioritized_leads.append({
                        "name": row['leads'],
                        "status": row['status'],
                        "sales_record": str(row['sales record of client']),
                        "priority": idx + 1
                    })
                
                # Display prioritized leads in a simple format
                st.write("### Lead Prioritization and Call Results:")
                output_df = prioritized.copy()
                output_df['Call Status'] = None
                for lead in prioritized_leads:
                    name = lead['name']
                    status = lead['status']
                    sales_record = lead['sales_record']
                    priority = lead['priority']
                    st.write(f"Priority {priority}: {name} (Status: {status}, Sales: {sales_record})")
                    
                    lead_row = output_df[output_df['leads'] == name]
                    if not lead_row.empty:
                        phone = lead_row.iloc[0]['phone number']
                        call_result = await manual_simulate_call(name, phone)
                        output_df.loc[output_df['leads'] == name, 'Call Status'] = call_result
                        st.write(f"- {call_result}")
                
                st.write("### Updated Lead Data:")
                st.dataframe(output_df)
                output_file = "updated_leads.xlsx"
                output_df.to_excel(output_file, index=False)
                st.success(f"Excel file updated with call status. Saved as {output_file}")
        else:
            st.error("No valid response detected.")
    except Exception as e:
        st.error(f"Error processing response: {str(e)}")