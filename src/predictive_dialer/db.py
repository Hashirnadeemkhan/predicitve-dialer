from typing import Optional
from sqlmodel import Field, SQLModel, create_engine, Session, select
from datetime import  datetime

valid_outcomes = ["Success", "No Answer", "Busy", "Rejected"]
class Calls(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    call_duration: Optional[int] = None
    call_outcomes: str = Field(default="No answer")
    time_of_call: Optional[datetime] = None
    agent_name: str = Field(default="unknown")
    lead_status: str
    customer_response: str
    follow_up_required: str = Field(default="no")
    notes: str

    def validating_call_outcomes(self):
        if self.call_outcomes not in valid_outcomes:
            raise ValueError(f"Invalid call_outcomes value: {self.call_outcomes} must be one of {valid_outcomes}")

database_url = "postgresql://dialerdb_owner:npg_ZqTU4WBX3mFv@ep-flat-bonus-a412iac9-pooler.us-east-1.aws.neon.tech/dialerdb?sslmode=require"

engine = create_engine(
    database_url,
    echo=True,
    pool_pre_ping=True,
    pool_recycle=300
)

SQLModel.metadata.create_all(engine)

def add_call(call: Calls):
    call.validating_call_outcomes()
    with Session(engine) as session:
        session.add(call)
        session.commit()
        session.refresh(call)
    return call

def display_calls():
    with Session(engine) as session:
        statement = select(Calls)
        results = session.exec(statement)
        calls = results.all()
        for call in calls:
            print(f"Call ID: {call.id}")
            print(f"Duration: {call.call_duration} minutes")
            print(f"Outcome: {call.call_outcomes}")
            print(f"Time of Call: {call.time_of_call}")
            print(f"Agent: {call.agent_name}")
            print(f"Lead Status: {call.lead_status}")   
            print(f"Customer Response: {call.customer_response}")
            print(f"Follow-up Required: {call.follow_up_required}")
            print("-" * 30)



if __name__ == "__main__":
    add_call()
    display_calls()