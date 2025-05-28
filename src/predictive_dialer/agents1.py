import os
import asyncio
import re
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from agents import set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv
import pandas as pd
import openpyxl

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

@function_tool
def excel_data():
    """Read and return all Excel data (with column names and full rows)"""
    try:
        import pandas as pd

        df = pd.read_excel(r'D:\piaic\resume_analyzer\Book1.xlsx')

        df = df.astype(str)

        return {
            "columns": df.columns.tolist(),             
            "data": df.to_dict(orient='records')        
        }

    except FileNotFoundError:
        return "Excel file not found"

@function_tool
async def simulate_call(name: str, phone: str) -> str:
    """Simulate a phone call to a lead"""
    if not phone or not isinstance(phone, str) or not re.match(r'^\d{10,}$', phone):
        return f"Skipping {name}: No valid phone number"
    print(f"Calling {name} at {phone}...")
    await asyncio.sleep(10)
    print(f"Call completed for {name}")
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
    lead_prioritizing_agent = Agent(
        name="csv Assistant",
        instructions='''
        You are expert in reading csv files, use the tool excel_data to read data and you will see some call center data, 
        your job is to see the status of every lead and prioritize them accordingly: 'Qualified' leads first, then 'New', then 'Pending'. 
        In the case If you see leads With the same status then call those leads first who has the highest sales record among those status. The higher the Sales Record the prior the lead should be. Do not break the rules of arrangement of call you have to give '1000%' focus on priority no matter what. And in the end return a call summary.
        Return a numbered list of lead names.''',
        tools=[excel_data],
        handoff_description="Csv file assistant, expert in sorting leads according to the status and sales record.",
        model=model
    )

    calling_agent = Agent(
        name="Calling agent",
        instructions='''
        You receive a prioritized list of lead names from the csv Assistant. For each lead, use the excel_data tool to find their phone number, 
        then use the simulate_call tool to simulate a call. Process leads in the given order and return a summary of call results.''',
        tools=[excel_data, simulate_call],
        handoff_description="Expert calling agent.",
        model=model
    )

    excel_column_agent = Agent(
        name="Excel Column Detector",
        instructions='''
    You're a data analyst. Use the tool `excel_data` to load the Excel file.

    Your job is to find:
    1. Which column contains the lead/customer names (values like "Ali", "Sarah", etc).
    2. Which column contains the phone numbers.
    3. Which column contains the lead status (values like "Qualified", "New", "Pending", etc).

    Return a JSON in this format:
    {
        "name_column": "...",
        "phone_column": "...",
        "status_column": "..."
    }
    ''',
        tools=[excel_data],
        model=model
    )


    manager_agent = Agent(
        name="Manager",
        instructions='''
    First, ask the Excel Column Detector to find correct column names for Name, Phone, and Status.

    Then pass that mapping to the Lead Prioritizing Agent so it can use the correct columns to prioritize.

    Then pass the prioritized list to the Calling Agent to simulate calls.

    All agents must use the correct column names detected in step 1.
    ''',
        handoffs=[excel_column_agent, lead_prioritizing_agent, calling_agent],
        model=model
    )

    

    result = Runner.run_streamed(manager_agent, input='Please sort the leads and simulate calls')
    prioritized_text = ""
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            prioritized_text += event.data.delta
            print(event.data.delta, end="", flush=True)

    lead_names = list(dict.fromkeys(re.findall(r'\d+\.\s*(\w+)\s*\(', prioritized_text)))
    if not lead_names:
        print("Error: No lead names found in prioritized list")
        return

    #save response of call outcomes in excel
    df = pd.read_excel("Book1.xlsx")
    wb = openpyxl.load_workbook("Book1.xlsx")
    sheet = wb.active
    last_col_index = sheet.max_column + 1
    sheet.cell(row=1, column=last_col_index).value = "Call Status"

    for i, lead in enumerate(lead_names, start=2):  
        lead_row = df[df['Name'].str.strip().str.lower() == lead.strip().lower()]
        if not lead_row.empty:
            phone = str(lead_row.iloc[0]['Phone Number'])
            name = lead_row.iloc[0]['Name']
            if not phone or not re.match(r'^\d{10,}$', phone):
                call_status = f"Skipping {name}: No valid phone number"
            else:
                print(f"Calling {name} at {phone}...")
                await asyncio.sleep(5)  # simulate call
                print(f"Call completed for {name}")
                call_status = f"Call completed for {name}"
            
            sheet.cell(row=i, column=last_col_index).value = call_status
        else:
            print(f"Lead {lead} not found in Excel file.")

    wb.save("Book1_updated.xlsx")
    print("\nExcel file updated with call status.")

if __name__ == "__main__":
    asyncio.run(main())