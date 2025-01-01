import queue
import threading
import torch
import logging
import time
from whisper_live.transcriber import WhisperModel

MODEL = "tiny"
NUM_MODELS = 5

class TranscriptionManager:
    def __init__(self):
        self.device, self.compute_type = self._get_device_type()
        self.model_pool = queue.Queue(maxsize=NUM_MODELS)
        self.lock = threading.Lock()

        logging.info(f'Creating {NUM_MODELS} models')
        for _ in range(NUM_MODELS):
            self.model_pool.put(self._create_model_instance())
        logging.info (f'Done model creation')

    def _create_model_instance(self):
        return WhisperModel(
            MODEL,
            device=self.device,
            compute_type=self.compute_type,
            local_files_only=False,
        )
    
    def _get_device_type(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            major, _ = torch.cuda.get_device_capability(device)
            compute_type = "float16" if major >= 7 else "float32"
        else:
            compute_type = "int8"
        return device, compute_type        

    def get_available_model(self):
        """
        Retrieves the first available model from the queue.
        Blocks if no models are available.
        """
        logging.info("getting a model")
        start_time = time.time()
        model = self.model_pool.get()
        end_time = time.time()
        logging.info(f"time spent retrieving model: {end_time - start_time:.2f}")
        return model

    def release_model(self, model):
        """
        Returns the model back to the queue after use.
        """
        with self.lock:
            self.model_pool.put(model)