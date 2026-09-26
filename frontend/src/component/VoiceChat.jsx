import { useEffect, useMemo, useRef, useState } from "react";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const ACCEPTED_FILES =
  ".pdf,.png,.jpg,.jpeg,.webp,.docx,.pptx,.xlsx,.csv,.txt,.md";

function createConversationId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }

  return `conversation-${Date.now()}`;
}

function formatMs(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }

  return `${Math.round(value)} ms`;
}

function sourceLabel(source) {
  const filename = source?.filename || "Document";
  const page =
    source?.page !== null && source?.page !== undefined
      ? ` · p. ${source.page}`
      : "";

  return `${filename}${page}`;
}

function sourceTypeLabel(source) {
  if (source?.retrieval_backend === "pageindex") {
    return "PageIndex";
  }

  if (source?.retrieval_backend === "qdrant") {
    return "Semantic search";
  }

  return source?.source_type || "Document";
}

export default function VoiceChat() {
  const [conversationId, setConversationId] = useState(
    createConversationId()
  );

  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState("");

  const [documents, setDocuments] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);

  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);

  const [recording, setRecording] = useState(false);
  const [processingVoice, setProcessingVoice] = useState(false);

  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const [health, setHealth] = useState(null);

  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const hasConversation = messages.length > 0;

  useEffect(() => {
    checkHealth();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, asking, processingVoice]);

  const documentCount = documents.length;

  const canSend = useMemo(
    () => query.trim().length > 0 && !asking && !processingVoice,
    [query, asking, processingVoice]
  );

  async function checkHealth() {
    try {
      const response = await fetch(`${API_URL}/health`);

      if (!response.ok) {
        throw new Error("Backend is unavailable.");
      }

      const data = await response.json();
      setHealth(data);
    } catch {
      setHealth({
        status: "offline",
      });
    }
  }

  function resetConversation() {
    setMessages([]);
    setQuery("");
    setError("");
    setNotice("");
    setConversationId(createConversationId());
  }

  function handleFileSelection(event) {
    const files = Array.from(event.target.files || []);

    setSelectedFiles(files);
    setError("");
    setNotice("");

    if (files.length > 0) {
      setNotice(
        `${files.length} file${files.length === 1 ? "" : "s"} selected.`
      );
    }
  }

  async function uploadDocuments() {
    if (selectedFiles.length === 0 || uploading) {
      return;
    }

    setUploading(true);
    setError("");
    setNotice("");

    try {
      const formData = new FormData();

      selectedFiles.forEach((file) => {
        formData.append("files", file);
      });

      const response = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Document upload failed.");
      }

      const uploaded = data.documents || [];

      setDocuments((current) => {
        const merged = [...current];

        uploaded.forEach((document) => {
          const existingIndex = merged.findIndex(
            (item) => item.document_id === document.document_id
          );

          if (existingIndex >= 0) {
            merged[existingIndex] = document;
          } else {
            merged.push(document);
          }
        });

        return merged;
      });

      setSelectedFiles([]);
      setNotice(
        `${uploaded.length} document${
          uploaded.length === 1 ? "" : "s"
        } indexed successfully.`
      );

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (err) {
      setError(err.message || "Document upload failed.");
    } finally {
      setUploading(false);
      checkHealth();
    }
  }

  async function askQuestion(event) {
    event?.preventDefault();

    const text = query.trim();

    if (!text || asking || processingVoice) {
      return;
    }

    setError("");
    setNotice("");
    setQuery("");

    const userMessage = {
      id: `${Date.now()}-user`,
      role: "user",
      content: text,
    };

    setMessages((current) => [...current, userMessage]);
    setAsking(true);

    try {
      const response = await fetch(`${API_URL}/rag`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: text,
          conversation_id: conversationId,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Question failed.");
      }

      setMessages((current) => [
        ...current,
        {
          id: `${Date.now()}-assistant`,
          role: "assistant",
          content: data.answer || "No answer returned.",
          grounded: data.grounded,
          sources: data.sources || [],
          latency: data.latency || null,
        },
      ]);
    } catch (err) {
      setError(err.message || "Unable to process the question.");
    } finally {
      setAsking(false);
    }
  }

  async function startRecording() {
    if (processingVoice || asking || recording) {
      return;
    }

    setError("");
    setNotice("");

    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Voice recording is not supported by this browser.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });

      streamRef.current = stream;
      chunksRef.current = [];

      let mimeType = "";

      if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
        mimeType = "audio/webm;codecs=opus";
      } else if (MediaRecorder.isTypeSupported("audio/webm")) {
        mimeType = "audio/webm";
      } else if (MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")) {
        mimeType = "audio/ogg;codecs=opus";
      }

      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);

      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onerror = () => {
        setError("The browser could not record the audio.");
        cleanupRecording();
      };

      recorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });

        cleanupRecording();

        if (blob.size === 0) {
          setError("No audio was recorded.");
          return;
        }

        await sendVoice(blob, recorder.mimeType);
      };

      recorder.start();
      setRecording(true);
    } catch (err) {
      console.error(err);
      cleanupRecording();

      if (err?.name === "NotAllowedError") {
        setError("Microphone permission was denied.");
      } else {
        setError("Unable to access the microphone.");
      }
    }
  }

  function stopRecording() {
    const recorder = mediaRecorderRef.current;

    if (!recorder) {
      return;
    }

    if (recorder.state === "recording") {
      recorder.stop();
    }

    setRecording(false);
    setProcessingVoice(true);
  }

  function cleanupRecording() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
    }

    streamRef.current = null;
    mediaRecorderRef.current = null;
    chunksRef.current = [];
    setRecording(false);
  }

  async function sendVoice(audioBlob, mimeType) {
    setProcessingVoice(true);
    setError("");
    setNotice("");

    const extension = mimeType?.includes("ogg") ? "ogg" : "webm";

    try {
      const formData = new FormData();

      formData.append(
        "audio",
        audioBlob,
        `recording.${extension}`
      );

      const response = await fetch(
        `${API_URL}/voice?conversation_id=${encodeURIComponent(conversationId)}`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Voice request failed.");
      }

      const transcript = data.transcript?.trim();

      if (!transcript) {
        throw new Error("No speech was detected.");
      }

      setMessages((current) => [
        ...current,
        {
          id: `${Date.now()}-user`,
          role: "user",
          content: transcript,
          voice: true,
        },
        {
          id: `${Date.now()}-assistant`,
          role: "assistant",
          content: data.answer || "No answer returned.",
          grounded: data.grounded,
          sources: data.sources || [],
          latency: data.latency || null,
        },
      ]);
    } catch (err) {
      setError(err.message || "Unable to process the voice question.");
    } finally {
      setProcessingVoice(false);
    }
  }

  function handleComposerKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();

      if (canSend) {
        askQuestion(event);
      }
    }
  }

  return (
    <main className="min-h-screen bg-[#edf2f7] text-slate-800 font-sans">
      <div className="mx-auto flex min-h-screen w-full max-w-[1600px] flex-col px-4 py-3 sm:px-6 lg:px-8">
        <header className="flex items-center justify-between rounded-lg border border-indigo-100 bg-white/80 px-6 py-3 shadow-sm backdrop-blur-md">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-indigo-600 text-white shadow-md shadow-indigo-200">
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
              </svg>
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-slate-900">
                DocIntel AI
              </h1>
              <p className="text-xs font-medium text-slate-500">
                Multi-Source Evidence Engine
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs font-medium">
            <div className="hidden items-center gap-2 rounded-full bg-slate-100/80 px-3 py-1 text-slate-600 sm:flex border border-slate-200/60">
              <span
                className={`h-2 w-2 rounded-full ${
                  health?.status === "healthy"
                    ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]"
                    : "bg-amber-400"
                }`}
              />
              {health?.status === "healthy" ? "Connected (FastAPI)" : "Offline"}
            </div>

            <button
              type="button"
              onClick={resetConversation}
              className="flex items-center gap-1.5 rounded-md bg-indigo-600 px-3.5 py-1.5 font-semibold text-white shadow-sm transition hover:bg-indigo-700 active:bg-indigo-800"
            >
              <span>+</span> New Conversation
            </button>
          </div>
        </header>

        <div className="grid flex-1 gap-4 py-4 lg:grid-cols-[320px_minmax(0,1fr)]">
          <aside className="flex flex-col rounded-xl border border-indigo-100/80 bg-white/90 shadow-sm">
            <div className="border-b border-slate-100 p-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                    Document Corpora
                  </h2>
                  <p className="mt-0.5 text-xs font-semibold text-indigo-600">
                    {documentCount} Active Documents
                  </p>
                </div>

                <span className="rounded bg-indigo-50 px-2 py-0.5 text-[11px] font-bold text-indigo-600">
                  Corpus
                </span>
              </div>

              <div className="mt-4 rounded-lg border-2 border-dashed border-indigo-200/70 bg-indigo-50/30 p-4 text-center transition hover:border-indigo-400">
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept={ACCEPTED_FILES}
                  onChange={handleFileSelection}
                  className="block w-full cursor-pointer text-xs text-slate-500 file:mr-2 file:rounded-md file:border-0 file:bg-indigo-600 file:px-2.5 file:py-1.5 file:text-xs file:font-semibold file:text-white hover:file:bg-indigo-700"
                />
              </div>

              {selectedFiles.length > 0 && (
                <div className="mt-3 rounded-lg border border-indigo-100 bg-indigo-50/50 p-3">
                  <p className="text-xs font-bold text-indigo-900">
                    Ready to upload
                  </p>

                  <div className="mt-2 max-h-24 overflow-y-auto space-y-1">
                    {selectedFiles.map((file) => (
                      <p
                        key={`${file.name}-${file.size}`}
                        className="truncate text-xs text-slate-600"
                        title={file.name}
                      >
                        • {file.name}
                      </p>
                    ))}
                  </div>

                  <button
                    type="button"
                    disabled={uploading}
                    onClick={uploadDocuments}
                    className="mt-3 w-full rounded-md bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {uploading ? "Indexing..." : "Upload and index"}
                  </button>
                </div>
              )}
            </div>

            <div className="max-h-[580px] flex-1 overflow-y-auto p-2">
              {documents.length === 0 ? (
                <div className="p-4 text-center text-xs leading-5 text-slate-400">
                  Upload files above to construct your indexed multi-source evidence base.
                </div>
              ) : (
                <div className="space-y-2">
                  {documents.map((document) => (
                    <div
                      key={document.document_id}
                      className="rounded-lg border border-slate-200/80 bg-white p-3 shadow-2xs hover:border-indigo-300 hover:shadow-xs transition"
                    >
                      <p
                        className="truncate text-xs font-bold text-slate-800"
                        title={document.filename}
                      >
                        📄 {document.filename}
                      </p>

                      <div className="mt-1 flex items-center justify-between text-[11px] font-medium text-slate-500">
                        <span className="rounded bg-slate-100 px-1.5 py-0.5 uppercase">
                          {document.file_type?.toUpperCase()}
                        </span>
                        <span>
                          {document.pages || 0} pages
                        </span>
                      </div>

                     
                    </div>
                  ))}
                </div>
              )}
            </div>
          </aside>

          <section className="flex flex-col overflow-hidden rounded-xl border border-indigo-100/80 bg-white shadow-sm">
            <div className="flex items-center justify-between border-b border-slate-100 bg-slate-50/50 px-6 py-3">
              <div>
                <h2 className="text-sm font-bold text-slate-800">
                  Synthesis Workspace
                </h2>
                <p className="text-xs text-slate-500">
                  Grounded multi-source document QA and intelligence context
                </p>
              </div>
              <span className="rounded-full bg-emerald-100 px-3 py-1 text-[11px] font-bold text-emerald-800 border border-emerald-200">
                Hybrid RAG Active
              </span>
            </div>

            <div className="flex-1 overflow-y-auto px-6 py-5 bg-[#f8fafc]">
              {!hasConversation ? (
                <div className="mx-auto flex min-h-[440px] max-w-2xl items-center justify-center">
                  <div className="w-full text-center">
                    <div className="mb-6 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-indigo-50 text-indigo-600 border border-indigo-100">
                      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21m7.5-18v1.5m0 15V21m-7.5-6h7.5" />
                      </svg>
                    </div>
                    <p className="text-base font-bold text-slate-800">
                      Query your indexed knowledge base
                    </p>
                    <p className="mt-1 text-xs text-slate-500 max-w-md mx-auto">
                      Select a prompt below or ask a natural question to retrieve verifiable sources and citations.
                    </p>

                    <div className="mt-6 grid gap-2.5 sm:grid-cols-2 text-left">
                      {[
                        "What was the 2025 revenue?",
                        "Compare the revenue across the reports.",
                        "What information is in the uploaded tables?",
                        "What does the document say about the roadmap?",
                      ].map((example) => (
                        <button
                          key={example}
                          type="button"
                          onClick={() => setQuery(example)}
                          className="rounded-lg border border-indigo-100/80 bg-white p-3.5 text-xs font-medium text-slate-700 shadow-2xs transition hover:border-indigo-400 hover:bg-indigo-50/20 active:scale-[0.99]"
                        >
                          💬 {example}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="mx-auto max-w-3xl space-y-6">
                  {messages.map((message) => (
                    <Message
                      key={message.id}
                      message={message}
                    />
                  ))}

                  {(asking || processingVoice) && (
                    <div className="flex gap-3">
                      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-indigo-600 text-[10px] font-bold text-white shadow-xs">
                        AI
                      </div>

                      <div className="rounded-xl border border-indigo-100 bg-indigo-50/50 px-4 py-3 shadow-2xs">
                        <div className="flex items-center gap-2 text-xs font-semibold text-indigo-700">
                          <span className="h-2 w-2 animate-ping rounded-full bg-indigo-500" />
                          {processingVoice
                            ? "Transcribing voice and searching..."
                            : "Searching multi-source documents..."}
                        </div>
                      </div>
                    </div>
                  )}

                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            {error && (
              <div className="border-t border-rose-200 bg-rose-50 px-6 py-2.5 text-xs font-semibold text-rose-700">
                ⚠️ {error}
              </div>
            )}

            {notice && (
              <div className="border-t border-indigo-100 bg-indigo-50/60 px-6 py-2 text-xs font-semibold text-indigo-800">
                ℹ️ {notice}
              </div>
            )}

            <form
              onSubmit={askQuestion}
              className="border-t border-slate-200/80 bg-white p-4"
            >
              <div className="mx-auto flex max-w-3xl items-center gap-2">
                <textarea
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  onKeyDown={handleComposerKeyDown}
                  rows={2}
                  placeholder="Ask a question about your uploaded documents..."
                  disabled={asking || processingVoice}
                  className="min-h-[48px] flex-1 resize-none rounded-lg border border-slate-200 bg-slate-50/50 p-3 text-xs font-medium text-slate-800 outline-none transition placeholder:text-slate-400 focus:border-indigo-500 focus:bg-white focus:ring-2 focus:ring-indigo-100 disabled:bg-slate-100"
                />

                <button
                  type="button"
                  onClick={
                    recording ? stopRecording : startRecording
                  }
                  disabled={asking || processingVoice}
                  aria-label={
                    recording ? "Stop recording" : "Start recording"
                  }
                  className={`flex h-[48px] w-[48px] shrink-0 items-center justify-center rounded-lg border shadow-2xs transition ${
                    recording
                      ? "border-rose-300 bg-rose-50 text-rose-600 animate-pulse"
                      : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50 hover:text-indigo-600"
                  } disabled:cursor-not-allowed disabled:opacity-50`}
                >
                  {recording ? (
                    <span className="h-3.5 w-3.5 rounded-xs bg-rose-600" />
                  ) : (
                    <MicrophoneIcon />
                  )}
                </button>

                <button
                  type="submit"
                  disabled={!canSend}
                  className="h-[48px] rounded-lg bg-indigo-600 px-6 text-xs font-bold text-white shadow-xs transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Ask
                </button>
              </div>

              <div className="mx-auto mt-2 flex max-w-3xl items-center justify-between text-[11px] text-slate-400">
                <span>
                  Press Enter to send · Shift + Enter for a new line
                </span>

                <span className="hidden font-mono sm:inline">
                  ID: {conversationId.slice(0, 8)}
                </span>
              </div>
            </form>
          </section>
        </div>
      </div>
    </main>
  );
}

function Message({ message }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%]">
          <div className="rounded-xl bg-indigo-600 px-4 py-3 text-xs leading-5 font-medium text-white shadow-sm">
            {message.content}
          </div>

          {message.voice && (
            <p className="mt-1 text-right text-[10px] font-semibold text-indigo-500">
              🎙️ Voice Query
            </p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3">
      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-indigo-600 text-[10px] font-bold text-white shadow-xs">
        AI
      </div>

      <div className="min-w-0 flex-1">
        <div className="rounded-xl border border-indigo-100/60 bg-white p-4 shadow-2xs text-xs leading-6 text-slate-800">
          {message.content}
        </div>

        <div className="mt-2.5 flex flex-wrap items-center gap-2">
          <span
            className={`rounded-md border px-2.5 py-0.5 text-[11px] font-bold ${
              message.grounded
                ? "border-emerald-300 bg-emerald-50 text-emerald-800"
                : "border-amber-300 bg-amber-50 text-amber-800"
            }`}
          >
            {message.grounded
              ? "✓ GROUNDED IN EVIDENCE"
              : "⚠️ INSUFFICIENT EVIDENCE"}
          </span>

          {message.latency && (
            <span className="font-mono text-[11px] text-slate-400">
              Total: {formatMs(message.latency.total_ms)}
            </span>
          )}
        </div>

        {message.sources?.length > 0 && (
          <div className="mt-3 overflow-hidden rounded-lg border border-slate-200 bg-white shadow-2xs">
            <div className="border-b border-slate-100 bg-slate-50/60 px-3.5 py-2">
              <p className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                Fused Evidence & Citations ({message.sources.length})
              </p>
            </div>

            <div className="divide-y divide-slate-100">
              {message.sources.map((source, index) => (
                <details
                  key={`${source.source_id || index}-${source.filename}-${source.page}`}
                  className="group px-3.5 py-2.5 transition hover:bg-indigo-50/30"
                >
                  <summary className="cursor-pointer list-none">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="truncate text-xs font-bold text-slate-700">
                          [{index + 1}] {sourceLabel(source)}
                        </p>

                        <p className="mt-0.5 text-[11px] text-slate-400">
                          {sourceTypeLabel(source)}
                          {source.section
                            ? ` · ${source.section}`
                            : ""}
                        </p>
                      </div>

                      <span className="shrink-0 rounded bg-indigo-50 px-2 py-0.5 text-[10px] font-semibold text-indigo-600 group-open:bg-indigo-100">
                        View Chunk
                      </span>
                    </div>
                  </summary>

                  <div className="mt-2.5 border-t border-slate-100 pt-2.5">
                    <p className="whitespace-pre-wrap rounded bg-slate-50/80 p-2.5 font-mono text-[11px] leading-5 text-slate-600 border border-slate-100">
                      {source.text || "No evidence text returned."}
                    </p>

                    {source.citation && (
                      <div className="mt-2 text-[10px] font-semibold text-indigo-500">
                        Source Ref:{" "}
                        {source.citation.document ||
                          source.filename ||
                          "document"}
                        {source.citation.page
                          ? ` (Page ${source.citation.page})`
                          : ""}
                      </div>
                    )}
                  </div>
                </details>
              ))}
            </div>
          </div>
        )}

        {message.latency && (
          <div className="mt-2 flex flex-wrap gap-x-3 text-[10px] font-mono text-slate-400">
            {typeof message.latency.retrieval_ms === "number" && (
              <span>
                Retrieval: {formatMs(message.latency.retrieval_ms)}
              </span>
            )}

            {typeof message.latency.generation_ms === "number" && (
              <span>
                LLM: {formatMs(message.latency.generation_ms)}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function MicrophoneIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      className="h-4 w-4"
      aria-hidden="true"
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
  );
}