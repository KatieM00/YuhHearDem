from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Source(BaseModel):
    date: str
    speaker: str
    videoId: str
    timestamp: int

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    response: str
    used_knowledge_graph: bool
    sources: Optional[List[Source]] = None

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Mock response with realistic parliamentary data
        response_text = "In the most recent parliamentary session, several key points were discussed regarding economic reforms. The Prime Minister emphasized digital transformation initiatives and support for small businesses through new tax incentives."
        
        return ChatResponse(
            response=response_text,
            used_knowledge_graph=True,
            sources=[
                Source(
                    date=datetime.now().strftime("%b %d, %Y"),
                    speaker="Parliamentary Economic Debate",
                    videoId="DUkW_SbdQOw",
                    timestamp=240
                )
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)