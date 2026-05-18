import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager 

# CUDA_VISIBLE_DEVICES=7 uvicorn rm_api_ArmoRM:app --host 0.0.0.0 --port 6007 --workers 1
# --- Configuration ---
MODEL_ID = "ArmoRM-Llama3-8B-v0.1" 
DTYPE = torch.bfloat16
DEVICE_MAP = 'auto' 
TRUST_REMOTE_CODE = True 
MAX_LENGTH = 4096

# Global variable to store the ArmoRMPipeline instance
rm_pipeline_instance: Optional['ArmoRMPipeline'] = None 

class ArmoRMPipeline:
    """
    Wrapper class for the Reward Model, used to compute scores and extract last-layer hidden states.
    """
    def __init__(self, model_id: str, device_map: str = "auto", torch_dtype=torch.bfloat16, 
                 truncation: bool = True, trust_remote_code: bool = False, max_length: int = 4096):
        
        if not torch.cuda.is_available() and device_map == 'auto':
             print("CUDA not available. Loading model to CPU.")
             device_map = 'cpu'

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            output_hidden_states=True 
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length
        print(f"Model and tokenizer loaded successfully. Device: {self.device}")

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Accepts a conversation history in OpenAI format and returns the score and last hidden state.
        """
        self.model.eval()
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True, 
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True) 
            score = outputs.score.item()
            
            hidden = outputs['hidden_state']
            hidden = torch.tensor([]) 

        return {
            "score": score,
            "hidden_state": hidden.squeeze().detach().cpu().tolist() 
        }

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    content: str
    role: str

class ScoreRequest(BaseModel):
    chat_history: list[ChatMessage] = Field(..., description="List of conversation messages")

class ScoreResponse(BaseModel):
    score: float = Field(..., description="Reward score from the model")
    hidden_state: Optional[List[float]] = None  
    status: str
    message: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events for model loading.
    """
    global rm_pipeline_instance
    
    try:
        print(f"Initializing ArmoRMPipeline, loading model: {MODEL_ID}...")
        
        rm_pipeline_instance = ArmoRMPipeline(
            model_id=MODEL_ID,
            device_map=DEVICE_MAP,
            torch_dtype=DTYPE,
            trust_remote_code=TRUST_REMOTE_CODE,
            max_length=MAX_LENGTH
        )
        
        yield 

    except Exception as e:
        print(f"Failed to load model: {e}")
        rm_pipeline_instance = None
        yield 
        
    finally:
        print("Application shutting down...")

app = FastAPI(title="Reward Model Scoring API", lifespan=lifespan)

@app.post("/score", response_model=ScoreResponse)
async def get_reward_score(request: ScoreRequest):
    """Accepts a conversation history and returns the reward score and last hidden state."""
    global rm_pipeline_instance
    
    if rm_pipeline_instance is None:
        return ScoreResponse(score=0.0, hidden_state=None, status="error", message="Reward model failed to load.")
        
    try:
        chat_dict_list = [msg.model_dump() for msg in request.chat_history] 
        
        result = rm_pipeline_instance(chat_dict_list)
        
        reward = result["score"]
        hidden_state = result["hidden_state"]
        
        return ScoreResponse(
            score=reward, 
            hidden_state=hidden_state, 
            status="success"
        )
        
    except Exception as e:
        print(f"Error processing request: {e}")
        return ScoreResponse(score=0.0, hidden_state=None, status="error", message=f"Request processing failed: {e}")

# --- Entry Point ---
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=6007)