import os
import sys
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import tempfile
import logging
from pydantic import BaseModel

# Add voicerestore and BigVGAN directories to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, 'BigVGAN'))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from BigVGAN import bigvgan
    from BigVGAN.meldataset import get_mel_spectrogram
    from model import OptimizedAudioRestorationModel
except ImportError as e:
    logger.error(f"Error importing modules. Ensure you are running from the project root and voicerestore submodule is correctly initialized. Missing module: {e.name}")
    sys.exit(1)


app = FastAPI(title="Voice Restore API")

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

logger.info(f"Using device: {device}")

# --- Pydantic Model for Request ---
class RestoreRequest(BaseModel):
    input_path: str
    output_path: str
    steps: int = 32
    cfg_strength: float = 0.5

# --- Model Loading ---
try:
    bigvgan_model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False).to(device)
    bigvgan_model.remove_weight_norm()

    def load_model(save_path):
        """Loads the OptimizedAudioRestorationModel."""
        optimized_model = OptimizedAudioRestorationModel(device=device, bigvgan_model=bigvgan_model)
        state_dict = torch.load(save_path, map_location=torch.device(device))
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        optimized_model.voice_restore.load_state_dict(state_dict, strict=True)
        optimized_model.eval()
        optimized_model.to(device)
        return optimized_model

    checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'voicerestore-1.1.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}. Please ensure you have downloaded the model and placed it in voicerestore/checkpoints/voicerestore-1.1.pth")
    
    optimized_model = load_model(checkpoint_path)
    logger.info("Voice restoration model loaded successfully.")

except Exception as e:
    logger.error(f"Failed to load the model: {e}", exc_info=True)
    # Allow server to start but endpoints will fail.
    optimized_model = None


def restore_audio_from_file(model, input_path, output_path, steps=32, cfg_strength=0.5):
    """Restores audio from an input file path and saves it to an output file path."""
    audio_tensor, sr = torchaudio.load(input_path)
    audio_tensor = audio_tensor.mean(dim=0, keepdim=True) if audio_tensor.dim() > 1 else audio_tensor
    
    with torch.inference_mode():
        if device == 'cuda':
            with torch.autocast(device_type=device):
                restored_wav = model(audio_tensor.to(device), steps=steps, cfg_strength=cfg_strength)
        else:
            restored_wav = model(audio_tensor.to(device), steps=steps, cfg_strength=cfg_strength)
        restored_wav = restored_wav.squeeze(0).float().cpu()
    
    torchaudio.save(output_path, restored_wav, model.target_sample_rate)


@app.post("/restore-audio")
async def restore_audio_endpoint(request: RestoreRequest):
    if not optimized_model:
        raise HTTPException(status_code=500, detail="Model is not loaded. Cannot process request.")

    try:
        if not os.path.exists(request.input_path):
            raise FileNotFoundError()

        logger.info(f"Processing audio file from: {request.input_path}")
        restore_audio_from_file(
            optimized_model, 
            request.input_path, 
            request.output_path, 
            steps=request.steps, 
            cfg_strength=request.cfg_strength
        )
        logger.info(f"Audio restored and saved to {request.output_path}")
        
        return {"status": "success", "message": f"Audio restored and saved to {request.output_path}"}

    except FileNotFoundError:
        logger.error(f"Input file not found: {request.input_path}")
        raise HTTPException(status_code=404, detail=f"Input file not found at {request.input_path}")
    except Exception as e:
        logger.error(f"Error during audio restoration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error restoring audio: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # The server will be started from srt_to_speech.py, but this allows direct execution for testing.
    uvicorn.run(app, host="0.0.0.0", port=8001) 