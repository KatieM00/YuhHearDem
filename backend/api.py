from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

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
    sources: Optional[List[Source]] = None

@app.post("/chat")
async def chat(request: ChatRequest):
    # For now, return a mock response
    return ChatResponse(
        response="This is a mock response from the backend server.",
        sources=[
            Source(
                date="Jan 15, 2025",
                speaker="Parliamentary Session",
                videoId="dQw4w9WgXcQ",
                timestamp=930
            )
        ]
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)