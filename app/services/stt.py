from pathlib import Path
from typing import Optional
from groq import Groq
from app.config import settings

SUPPORTED_AUDIO_EXTENSIONS = {
    ".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"
}
MAX_CHUNK_BYTES = 25 * 1024 * 1024  # 25 MB Groq free-tier attachment limit

DEFAULT_CLINICAL_PROMPT = (
    "Doctor-patient clinical consultation. Medical terminology, symptoms, "
    "vital signs, diagnoses, and medication dosages."
)


class AudioValidationError(ValueError):
    """Raised when an audio chunk is empty, oversized, or an unsupported format."""


def validate_audio_chunk(filename: str, audio_bytes: bytes) -> None:
    if not audio_bytes:
        raise AudioValidationError("Audio chunk is empty.")
    if len(audio_bytes) > MAX_CHUNK_BYTES:
        raise AudioValidationError(
            f"Audio chunk ({len(audio_bytes)} bytes) exceeds 25 MB limit."
        )
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        raise AudioValidationError(
            f"Unsupported audio format '{ext}'. Supported: {sorted(SUPPORTED_AUDIO_EXTENSIONS)}"
        )


def append_transcript_chunk(existing_transcript: str, new_chunk_text: str) -> str:
    """Cleanly concatenates sequential ~30s chunk transcripts."""
    cleaned_existing = existing_transcript.strip()
    cleaned_new = new_chunk_text.strip()
    if not cleaned_new:
        return cleaned_existing
    if not cleaned_existing:
        return cleaned_new
    return f"{cleaned_existing} {cleaned_new}"


class GroqSTTService:
    def __init__(self, client: Optional[Groq] = None, model: Optional[str] = None):
        self.client = client or Groq(api_key=settings.groq_api_key)
        self.model = model or settings.groq_stt_model

    def transcribe_chunk(
        self,
        filename: str,
        audio_bytes: bytes,
        previous_transcript: str = "",
        language: str = "en",
    ) -> str:
        """
        Transcribes a single ~30s audio chunk using Groq Whisper Large V3.
        Uses trailing context from previous_transcript to preserve medical terms across boundaries.
        """
        validate_audio_chunk(filename, audio_bytes)

        context_prompt = DEFAULT_CLINICAL_PROMPT
        if previous_transcript.strip():
            tail = previous_transcript.strip()[-200:]
            context_prompt = f"{DEFAULT_CLINICAL_PROMPT} Context: {tail}"

        transcription = self.client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model=self.model,
            prompt=context_prompt,
            response_format="text",
            language=language,
            temperature=0.0,
        )

        # response_format="text" returns a plain str (or an object with .text depending on SDK version)
        if isinstance(transcription, str):
            return transcription.strip()
        return getattr(transcription, "text", str(transcription)).strip()