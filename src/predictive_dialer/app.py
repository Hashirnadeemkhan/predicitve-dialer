import os
import re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import time
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("Gemini API key not found. Please set GEMINI_API_KEY in your .env file.")
    st.stop()
genai.configure(api_key=gemini_api_key)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')

# Streamlit page configuration
st.set_page_config(page_title="Lead Call Simulator with Sales Agent", layout="wide")
st.title("📞 Lead Prioritization & Call Simulation with Sales Agent")

# Session state for file upload and chat history
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to clean phone numbers
def clean_phone_number(phone):
    if pd.isna(phone):
        return None
    phone_str = str(phone).replace(',', '').replace(' ', '').replace('-', '')
    return phone_str if phone_str.isdigit() else None

# Function to validate phone numbers
def is_valid_phone_number(phone):
    if not phone:
        return False
    return bool(re.match(r'^\d{10,}$', phone))

# Sales script template
sales_script = """
Hello [NAME], my name is Alex from SecureTech Solutions. I’m calling to offer you our Smart Home Security System, designed to protect your home with advanced features like 24/7 monitoring and smart alerts. This system is currently available at a special discount of 20% off, saving you $200!

I see your phone number is [PHONE]. Is this a good time to discuss how this can benefit you? 
- If yes, I’ll explain the features and offer a limited-time deal.
- If no, I can schedule a follow-up call at a time that suits you.

[AGENT_RESPONSE: Based on their response, proceed with the pitch or schedule a follow-up.]
If interested, the offer includes a free installation worth $150. Would you like to proceed with the purchase today?
- If yes, I’ll guide you through the next steps.
- If no, I’ll note your interest for a future offer.
"""

# Function to generate agent conversation using Gemini API with sales script
def generate_agent_conversation(name, phone):
    conversation = []
    # Replace placeholders in the script
    script = sales_script.replace("[NAME]", name).replace("[PHONE]", str(phone) if phone else "not provided")
    
    # Initial prompt to Gemini
    prompt = f"Follow this sales script strictly: {script}. The lead's name is {name}, and the phone number is {phone}. Generate the agent's initial response and handle the conversation as if the lead says 'Yes, this is a good time.' Then ask if they want to proceed with the purchase."
    try:
        response = model.generate_content(prompt)
        conversation.append(f"Agent: {response.text}")
        
        # Follow-up prompt for purchase decision
        follow_up_prompt = f"Based on the previous response, the lead said 'Yes, this is a good time.' Continue the script and ask {name} if they would like to proceed with the purchase of the Smart Home Security System today."
        follow_up_response = model.generate_content(follow_up_prompt)
        conversation.append(f"Agent: {follow_up_response.text}")
        
        # Mock user response
        conversation.append(f"User: Yes, I want to proceed.")
        conversation.append(f"Agent: Excellent! I’ll guide you through the purchase process. Please provide your email address, or I can send a link to complete the order. Thank you, {name}!")
    except Exception as e:
        conversation.append(f"Agent: Error generating response: {str(e)}")
    return conversation

# Simulate a phone call (synchronous for now)
def simulate_call(name: str, phone: str) -> str:
    st.write(f"Calling {name} at {phone}...")
    time.sleep(2)
    st.write(f"Call completed for {name}")
    return f"Call completed for {name}"

# Main processing function
def process_leads():
    # Load the Excel file
    try:
        df = pd.read_excel(st.session_state.uploaded_file, engine='openpyxl')
        st.success("✅ File uploaded successfully.")
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return

    # Normalize column names
    df.columns = [col.strip().lower() for col in df.columns]

    # Check for required columns
    required_columns = ['leads', 'phone number', 'status', 'sales record of client']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        return

    # Clean phone numbers
    df['cleaned_phone'] = df['phone number'].apply(clean_phone_number)

    # Prioritize leads
    status_order = {'qualified': 0, 'new': 1, 'pending': 2}
    df['status_order'] = df['status'].str.lower().map(status_order).fillna(3)
    prioritized_df = df.sort_values(by=['status_order', 'sales record of client'], 
                                    ascending=[True, False])
    prioritized_df = prioritized_df.drop(columns=['status_order'])

    # Add priority column
    prioritized_df['priority'] = range(1, len(prioritized_df) + 1)

    # Display prioritized leads and simulate agent conversation
    st.write("### Lead Prioritization and Call Results:")
    output_df = prioritized_df.copy()
    output_df['call status'] = None

    # Process each lead
    for idx, row in output_df.iterrows():
        name = row['leads']
        status = row['status']
        sales_record = row['sales record of client']
        priority = row['priority']
        phone = row['cleaned_phone']

        st.write(f"Priority {priority}: {name} (Status: {status}, Sales: {sales_record})")

        # Generate agent conversation with Gemini and sales script
        conversation = generate_agent_conversation(name, phone)
        for message in conversation:
            st.session_state.chat_history.append(message)
            st.write(message)

        # Simulate call if phone is valid
        if phone and is_valid_phone_number(phone):
            call_result = simulate_call(name, phone)
        else:
            call_result = f"Skipping {name}: No valid phone number"
        output_df.loc[idx, 'call status'] = call_result
        st.write(f"- {call_result}")

    # Display chat history
    st.write("### Agent Conversation History:")
    for msg in st.session_state.chat_history:
        st.text(msg)

    # Display updated lead data
    st.write("### Updated Lead Data:")
    output_df = output_df.drop(columns=['cleaned_phone'])
    st.dataframe(output_df)

    # Save the updated Excel file
    output_file = "updated_leads.xlsx"
    output_df.to_excel(output_file, index=False)
    st.success(f"Excel file updated with prioritization and call status. Saved as {output_file}")

# File upload UI
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file

if st.button("Start Lead Prioritization and Calling"):
    if st.session_state.uploaded_file is None:
        st.error("Please upload a valid Excel file first.")
    else:
        st.session_state.chat_history = []  # Clear chat history
        with st.spinner("Processing leads and simulating calls..."):
            process_leads()

if __name__ == "__main__":
    pass