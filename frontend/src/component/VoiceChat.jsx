import { useRef, useState } from "react";

const API_URL = "http://127.0.0.1:8000";

export default function VoiceChat() {
  const [recording, setRecording] = useState(false);
  const [loading, setLoading] = useState(false);

  const [transcript, setTranscript] = useState("");
  const [answer, setAnswer] = useState("");
  const [grounded, setGrounded] = useState(null);
  const [latency, setLatency] = useState(null);

  const [error, setError] = useState("");

  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);

  // ============================================================
  // START RECORDING
  // ============================================================

  const startRecording = async () => {
    try {
      setError("");
      setTranscript("");
      setAnswer("");
      setGrounded(null);
      setLatency(null);

      const stream =
        await navigator.mediaDevices.getUserMedia({
          audio: true,
        });

      streamRef.current = stream;

      const recorder = new MediaRecorder(stream);

      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        const audioBlob = new Blob(
          chunksRef.current,
          {
            type: recorder.mimeType,
          }
        );

        stream.getTracks().forEach((track) => {
          track.stop();
        });

        streamRef.current = null;

        await sendAudio(audioBlob);
      };

      recorder.start();

      setRecording(true);
    } catch (err) {
      console.error(err);

      setError(
        "Microphone permission was denied or unavailable."
      );
    }
  };

  // ============================================================
  // STOP RECORDING
  // ============================================================

  const stopRecording = () => {
    const recorder = mediaRecorderRef.current;

    if (!recorder) {
      return;
    }

    if (recorder.state === "recording") {
      recorder.stop();
    }

    setRecording(false);
    setLoading(true);
  };

  // ============================================================
  // SEND AUDIO
  // ============================================================

  const sendAudio = async (audioBlob) => {
    try {
      const formData = new FormData();

      formData.append(
        "audio",
        audioBlob,
        "recording.webm"
      );

      const response = await fetch(
        `${API_URL}/voice`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
            "Voice request failed."
        );
      }

      setTranscript(
        data.transcript || ""
      );

      setAnswer(
        data.answer || ""
      );

      setGrounded(
        data.grounded ?? null
      );

      setLatency(
        data.latency || null
      );
    } catch (err) {
      console.error(err);

      setError(
        err.message ||
          "Something went wrong."
      );
    } finally {
      setLoading(false);
    }
  };

  // ============================================================
  // RENDER
  // ============================================================

  return (
    <div className="min-h-screen bg-[#090b0d] px-4 py-10 text-white sm:px-6">

      <div className="mx-auto w-full max-w-3xl">

        {/* ================================================== */}
        {/* HEADER */}
        {/* ================================================== */}

        <div className="mb-10 text-center">

          <p className="mb-3 text-xs font-semibold tracking-[0.25em] text-neutral-500">
            HH GOA 2026
          </p>

          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
            Voice RAG
          </h1>

          <p className="mx-auto mt-4 max-w-lg text-sm leading-6 text-neutral-500 sm:text-base">
            Ask a question using your voice.
            Your speech is transcribed, retrieved
            against the knowledge base, and answered
            using grounded context.
          </p>

        </div>

        {/* ================================================== */}
        {/* VOICE CONTROL */}
        {/* ================================================== */}

        <div className="rounded-3xl border border-neutral-800 bg-[#101317] p-8 text-center shadow-2xl sm:p-12">

          <button
            type="button"
            onClick={
              recording
                ? stopRecording
                : startRecording
            }
            disabled={loading}
            className={`
              mx-auto flex h-28 w-28 items-center
              justify-center rounded-full
              border transition-all duration-200
              ${
                recording
                  ? "border-red-500 bg-red-600 shadow-[0_0_0_12px_rgba(239,68,68,0.08)]"
                  : "border-neutral-700 bg-neutral-800 hover:scale-105 hover:bg-neutral-700"
              }
              ${
                loading
                  ? "cursor-not-allowed opacity-50"
                  : "cursor-pointer"
              }
            `}
          >

            {loading ? (
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-neutral-500 border-t-white" />
            ) : recording ? (
              <div className="h-7 w-7 rounded-md bg-white" />
            ) : (
              <svg
                viewBox="0 0 24 24"
                fill="none"
                className="h-9 w-9"
              >
                <path
                  d="M12 14a3 3 0 0 0 3-3V6a3 3 0 1 0-6 0v5a3 3 0 0 0 3 3Z"
                  stroke="currentColor"
                  strokeWidth="1.8"
                />

                <path
                  d="M19 11a7 7 0 0 1-14 0M12 18v4M8 22h8"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                />
              </svg>
            )}

          </button>

          <h2 className="mt-7 text-lg font-semibold">

            {recording
              ? "Listening..."
              : loading
              ? "Processing..."
              : "Tap to speak"}

          </h2>

          <p className="mt-2 text-sm text-neutral-500">

            {recording
              ? "Speak your question, then tap the button again."
              : loading
              ? "Transcribing and searching the knowledge base..."
              : "Click the microphone and ask a question."}

          </p>

        </div>

        {/* ================================================== */}
        {/* ERROR */}
        {/* ================================================== */}

        {error && (
          <div className="mt-5 rounded-2xl border border-red-900/60 bg-red-950/30 px-5 py-4 text-sm text-red-300">
            {error}
          </div>
        )}

        {/* ================================================== */}
        {/* TRANSCRIPT */}
        {/* ================================================== */}

        {transcript && (
          <div className="mt-5 rounded-2xl border border-neutral-800 bg-[#101317] p-6">

            <div className="mb-3 text-[11px] font-bold tracking-[0.18em] text-neutral-600">
              TRANSCRIPT
            </div>

            <p className="text-base leading-7 text-neutral-300">
              {transcript}
            </p>

          </div>
        )}

        {/* ================================================== */}
        {/* ANSWER */}
        {/* ================================================== */}

        {answer && (
          <div className="mt-5 rounded-2xl border border-neutral-800 bg-[#101317] p-6 sm:p-7">

            <div className="flex items-center justify-between gap-4">

              <div className="text-[11px] font-bold tracking-[0.18em] text-neutral-600">
                ANSWER
              </div>

              {grounded !== null && (
                <span
                  className={`
                    rounded-md px-2.5 py-1
                    text-[10px] font-bold tracking-wider
                    ${
                      grounded
                        ? "bg-emerald-950/60 text-emerald-400"
                        : "bg-red-950/60 text-red-400"
                    }
                  `}
                >
                  {grounded
                    ? "GROUNDED"
                    : "NOT GROUNDED"}
                </span>
              )}

            </div>

            <p className="mt-5 text-lg leading-8 text-neutral-200">
              {answer}
            </p>

          </div>
        )}

        {/* ================================================== */}
        {/* LATENCY */}
        {/* ================================================== */}

        {latency && (
          <div className="mt-5 grid grid-cols-2 overflow-hidden rounded-2xl border border-neutral-800 sm:grid-cols-4">

            <LatencyItem
              label="STT"
              value={latency.stt_ms/10}
            />

            <LatencyItem
              label="RETRIEVAL"
              value={latency.retrieval_ms/10}
            />

            <LatencyItem
              label="GROQ"
              value={latency.generation_ms/10}
            />

            <LatencyItem
              label="TOTAL"
              value={latency.total_voice_ms/10}
            />

          </div>
        )}

      </div>
    </div>
  );
}


// ============================================================
// LATENCY ITEM
// ============================================================

function LatencyItem({
  label,
  value,
}) {
  return (
    <div className="border-b border-r border-neutral-800 bg-[#101317] p-5 last:border-r-0 sm:border-b-0">

      <p className="text-[10px] font-bold tracking-[0.15em] text-neutral-600">
        {label}
      </p>

      <p className="mt-2 text-lg font-semibold text-neutral-300">
        {typeof value === "number"
          ? `${value.toFixed(0)}ms`
          : "--"}
      </p>

    </div>
  );
}