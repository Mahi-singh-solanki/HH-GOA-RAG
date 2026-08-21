from __future__ import annotations

import os
import tempfile
from pathlib import Path

from sarvamai import SarvamAI

from app.config import get_settings


class STTService:

    def __init__(self):

        settings = get_settings()

        self.client = SarvamAI(
            api_subscription_key=(
                settings.sarvam_api_key
            )
        )

    # ========================================================
    # TRANSCRIBE
    # ========================================================

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
    ) -> dict:

        if not audio_bytes:

            raise ValueError(
                "Empty audio file."
            )

        suffix = Path(
            filename
        ).suffix

        if not suffix:
            suffix = ".wav"

        temp_path = None

        try:

            # ------------------------------------------------
            # Save uploaded audio temporarily
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp:

                temp.write(
                    audio_bytes
                )

                temp_path = temp.name

            # ------------------------------------------------
            # Sarvam STT
            # ------------------------------------------------

            with open(
                temp_path,
                "rb",
            ) as audio_file:

                response = (
                    self.client.speech_to_text.transcribe(
                        file=audio_file
                    )
                )

            # ------------------------------------------------
            # Extract transcript
            # ------------------------------------------------

            transcript = getattr(
                response,
                "transcript",
                None,
            )

            if transcript is None:

                # Some SDK responses may expose
                # dictionary-style data.

                if isinstance(
                    response,
                    dict,
                ):

                    transcript = (
                        response.get(
                            "transcript"
                        )
                    )

            if not transcript:

                raise RuntimeError(
                    "Sarvam returned an "
                    "empty transcript."
                )

            return {
                "success": True,

                "text": transcript.strip(),
            }

        finally:

            # ------------------------------------------------
            # Remove temporary file
            # ------------------------------------------------

            if temp_path:

                try:

                    os.remove(
                        temp_path
                    )

                except OSError:

                    pass


# ============================================================
# SINGLETON
# ============================================================

_stt_service = None


def get_stt_service() -> STTService:

    global _stt_service

    if _stt_service is None:

        _stt_service = STTService()

    return _stt_service