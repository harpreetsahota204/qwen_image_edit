import React, { useCallback, useEffect, useRef, useState } from "react";
import { useRecoilValue } from "recoil";
import { modalGroupSlice as fosModalGroupSlice } from "@fiftyone/state";
import { useOperatorExecutor } from "@fiftyone/operators";
import { usePanelClient } from "./hooks/usePanelClient";
import { Turn, TurnImage, EditResult } from "./types";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PLUGIN_NAME = "@harpreetsahota/qwen_image_edit";
const QWEN_MODEL_ID = "drbaph/Qwen-Image-Edit-2511-FP8";

function makeMediaUrl(filepath: string): string {
  return `${window.location.origin}/media?filepath=${encodeURIComponent(filepath)}`;
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const S = {
  root: {
    display: "flex",
    flexDirection: "column" as const,
    height: "100%",
    overflow: "hidden",
    fontFamily: "var(--fo-fontFamily-body, sans-serif)",
    fontSize: 13,
    color: "#d0d0d0",
    boxSizing: "border-box" as const,
  },
  topBar: {
    display: "flex",
    gap: 8,
    padding: "8px 12px",
    borderBottom: "1px solid #3a3a3a",
    flexShrink: 0,
    alignItems: "center",
    justifyContent: "flex-end" as const,
  },
  advancedToggle: {
    background: "none",
    border: "none",
    color: "#777",
    cursor: "pointer",
    fontSize: 11,
    padding: "2px 4px",
    flexShrink: 0,
    textDecoration: "underline",
    lineHeight: 1.4,
  },
  advancedPanel: {
    padding: "8px 12px",
    borderBottom: "1px solid #2a2a2a",
    background: "#111",
    display: "flex",
    flexDirection: "column" as const,
    gap: 6,
    flexShrink: 0,
  },
  advancedRow: {
    display: "flex",
    gap: 8,
    alignItems: "center",
    flexWrap: "wrap" as const,
  },
  advancedLabel: {
    fontSize: 11,
    color: "#777",
    flexShrink: 0,
    minWidth: 110,
  },
  advancedInput: {
    flex: 1,
    minWidth: 80,
    background: "#1a1a1a",
    color: "#d0d0d0",
    border: "1px solid #3a3a3a",
    borderRadius: 4,
    padding: "4px 8px",
    fontSize: 12,
    outline: "none",
  },
  scroll: {
    flex: 1,
    overflowY: "auto" as const,
    padding: "10px 12px",
  },
  turnLabel: {
    fontSize: 12,
    color: "#888",
    fontStyle: "italic",
    marginBottom: 4,
    padding: "0 2px",
    display: "flex",
    justifyContent: "space-between" as const,
    alignItems: "center",
  },
  turnTime: {
    fontSize: 11,
    color: "#666",
    fontStyle: "normal",
  },
  imgWrap: {
    position: "relative" as const,
    marginBottom: 8,
    lineHeight: 0,
  },
  img: {
    width: "100%",
    display: "block",
    borderRadius: 4,
    objectFit: "contain" as const,
    maxHeight: 420,
  },
  badge: {
    position: "absolute" as const,
    bottom: 6,
    left: 6,
    background: "rgba(0,0,0,0.6)",
    color: "#fff",
    fontSize: 11,
    padding: "2px 6px",
    borderRadius: 3,
    lineHeight: 1.5,
    pointerEvents: "none" as const,
  },
  imgIndexBadge: {
    position: "absolute" as const,
    top: 6,
    left: 6,
    background: "rgba(0,0,0,0.6)",
    color: "#ccc",
    fontSize: 11,
    padding: "2px 6px",
    borderRadius: 3,
    lineHeight: 1.5,
    pointerEvents: "none" as const,
  },
  imgActions: {
    position: "absolute" as const,
    bottom: 6,
    right: 6,
    display: "flex",
    gap: 4,
    alignItems: "center",
  },
  imgActionBtn: (muted: boolean): React.CSSProperties => ({
    background: "rgba(0,0,0,0.6)",
    border: "1px solid rgba(255,255,255,0.15)",
    borderRadius: 4,
    color: muted ? "#888" : "#d0d0d0",
    cursor: muted ? "not-allowed" : "pointer",
    padding: "4px 7px",
    fontSize: 14,
    lineHeight: 1,
    display: "flex",
    alignItems: "center",
    gap: 4,
  }),
  loading: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "6px 0",
    color: "#888",
    fontSize: 13,
  },
  spinner: {
    width: 14,
    height: 14,
    border: "2px solid #444",
    borderTop: "2px solid #aaa",
    borderRadius: "50%",
    display: "inline-block",
    animation: "qwenEditSpin 0.7s linear infinite",
  } as React.CSSProperties,
  errorBox: {
    padding: "8px 10px",
    background: "#6b2020",
    color: "#ffc0c0",
    borderRadius: 4,
    marginBottom: 8,
    fontSize: 12,
  },
  warnBox: {
    padding: "8px 10px",
    background: "#4a3a10",
    color: "#ffd580",
    borderRadius: 4,
    marginBottom: 8,
    fontSize: 12,
  },
  bottomBar: {
    padding: "8px 12px",
    borderTop: "1px solid #3a3a3a",
    flexShrink: 0,
  },
  inputRow: {
    display: "flex",
    gap: 6,
    alignItems: "center",
  },
  input: {
    flex: 1,
    background: "#1a1a1a",
    color: "#d0d0d0",
    border: "1px solid #4a4a4a",
    borderRadius: 4,
    padding: "6px 10px",
    fontSize: 13,
    outline: "none",
    minWidth: 0,
  },
  sendBtn: (enabled: boolean): React.CSSProperties => ({
    background: enabled ? "#444" : "#2a2a2a",
    color: enabled ? "#d0d0d0" : "#555",
    border: "1px solid #4a4a4a",
    borderRadius: 4,
    padding: "6px 10px",
    cursor: enabled ? "pointer" : "not-allowed",
    fontSize: 15,
    lineHeight: 1,
    flexShrink: 0,
  }),
  empty: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    color: "#666",
    padding: 24,
    textAlign: "center" as const,
    fontSize: 13,
  },
};

// ---------------------------------------------------------------------------
// Icons
// ---------------------------------------------------------------------------

const SaveIcon: React.FC = () => (
  <svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor">
    <path d="M17 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V7l-4-4zm2 16H5V5h11.17L19 7.83V19zm-7-7c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3zM6 6h9v4H6z" />
  </svg>
);

const TrashIcon: React.FC = () => (
  <svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor">
    <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z" />
  </svg>
);

const SourceIcon: React.FC = () => (
  <svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor">
    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm0-12.5c-2.49 0-4.5 2.01-4.5 4.5S9.51 16.5 12 16.5s4.5-2.01 4.5-4.5S14.49 7.5 12 7.5zm0 7c-1.38 0-2.5-1.12-2.5-2.5S10.62 9.5 12 9.5s2.5 1.12 2.5 2.5S13.38 14.5 12 14.5z" />
  </svg>
);

// ---------------------------------------------------------------------------
// Spin keyframe
// ---------------------------------------------------------------------------

let spinInjected = false;
function ensureSpinKeyframe() {
  if (spinInjected) return;
  spinInjected = true;
  const el = document.createElement("style");
  el.textContent = "@keyframes qwenEditSpin { to { transform: rotate(360deg); } }";
  document.head.appendChild(el);
}

// ---------------------------------------------------------------------------
// Session persistence
// ---------------------------------------------------------------------------

interface CachedSession {
  turns: Turn[];
  sourceRef: { turnIdx: number; imageIdx: number };
}

const CACHE_PREFIX = "qwenEdit:";

function loadSessionValidated(sampleId: string, currentFilepath: string): CachedSession | null {
  try {
    const raw = sessionStorage.getItem(CACHE_PREFIX + sampleId);
    if (!raw) return null;
    const session = JSON.parse(raw) as CachedSession;
    if (!session.turns.length || session.turns[0].images?.[0]?.filepath !== currentFilepath) return null;
    return session;
  } catch {
    return null;
  }
}

function saveSession(sampleId: string, session: CachedSession): void {
  try {
    sessionStorage.setItem(CACHE_PREFIX + sampleId, JSON.stringify(session));
  } catch {
    // sessionStorage full or unavailable — silently skip
  }
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface PanelData {
  original_filepath?: string;
  sample_id?: string;
  label_fields?: string[];
  dataset_is_grouped?: boolean;
}

interface ImageEditPanelProps {
  data?: PanelData;
  schema?: {
    view?: {
      run_edit?: string;
      update_slice?: string;
      delete_turn?: string;
      get_pipeline_status?: string;
    };
  };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const QwenImageEditPanel: React.FC<ImageEditPanelProps> = ({ data, schema }) => {
  const runEditUri           = schema?.view?.run_edit            ?? "";
  const updateSliceUri       = schema?.view?.update_slice        ?? "";
  const deleteTurnUri        = schema?.view?.delete_turn         ?? "";
  const getPipelineStatusUri = schema?.view?.get_pipeline_status ?? "";

  const activeModalSlice = useRecoilValue<string | null>(fosModalGroupSlice);

  const [activeFilepath, setActiveFilepath]   = useState<string>("");
  const [activeSampleId, setActiveSampleId]   = useState<string>("");
  const [turns, setTurns]                     = useState<Turn[]>([]);
  // sourceRef tracks which specific image is the active edit source
  const [sourceRef, setSourceRef]             = useState<{ turnIdx: number; imageIdx: number }>({ turnIdx: 0, imageIdx: 0 });
  const [prompt, setPrompt]                   = useState<string>("");
  const [isLoading, setIsLoading]             = useState<boolean>(false);
  const [errorMsg, setErrorMsg]               = useState<string | null>(null);
  const [warnMsg, setWarnMsg]                 = useState<string | null>(null);
  const [pipelineStatus, setPipelineStatus]   = useState<string>("");

  // Advanced parameters
  const [showAdvanced, setShowAdvanced]               = useState<boolean>(false);
  const [negativePrompt, setNegativePrompt]           = useState<string>("");
  const [numSteps, setNumSteps]                       = useState<string>("");
  const [trueCfgScale, setTrueCfgScale]               = useState<string>("");
  const [seed, setSeed]                               = useState<string>("51");
  const [numImagesPerPrompt, setNumImagesPerPrompt]   = useState<string>("1");
  const [availableLabelFields, setAvailableLabelFields] = useState<string[]>([]);
  const [selectedLabelFields, setSelectedLabelFields]   = useState<Set<string>>(new Set());

  // Live elapsed-time counter
  const [elapsedSec, setElapsedSec]     = useState<number>(0);
  const editStartRef                    = useRef<number | null>(null);

  // Per-image saving / deleting / saved state (keys are "turnIdx:imageIdx")
  const [savingImages, setSavingImages]   = useState<Set<string>>(new Set());
  const [savedImages, setSavedImages]     = useState<Set<string>>(new Set());
  const [deletingImages, setDeletingImages] = useState<Set<string>>(new Set());
  const [saveErrorMsg, setSaveErrorMsg]   = useState<string | null>(null);

  const { runEdit, updateSlice, deleteTurn, getPipelineStatus } =
    usePanelClient(runEditUri, updateSliceUri, deleteTurnUri, getPipelineStatusUri);
  const saveExecutor = useOperatorExecutor(`${PLUGIN_NAME}/save_edited_images`);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Captures the prompt and expected turn index at submit time so the
  // polling effect can build the Turn when the result arrives.
  const pendingEditRef = useRef<{
    prompt: string;
    negativePrompt: string;
    expectedTurnIdx: number;
  } | null>(null);

  // ── Sync from Python lifecycle props ───────────────────────────────────────
  const prevDataSampleIdRef = useRef<string>("");
  useEffect(() => {
    const newId   = data?.sample_id         ?? "";
    const newPath = data?.original_filepath ?? "";
    if (!newId || !newPath) return;
    if (newId === prevDataSampleIdRef.current) return;
    prevDataSampleIdRef.current = newId;
    setActiveSampleId(newId);
    setActiveFilepath(newPath);
    if (Array.isArray(data?.label_fields)) setAvailableLabelFields(data.label_fields);
    prevSliceRef.current = null;
  }, [data?.sample_id, data?.original_filepath]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Slice tab detection ────────────────────────────────────────────────────
  const prevSliceRef = useRef<string | null>(null);
  useEffect(() => {
    if (!activeModalSlice || !data?.sample_id) return;
    if (activeModalSlice === prevSliceRef.current) return;
    prevSliceRef.current = activeModalSlice;
    updateSlice({ slice: activeModalSlice }).then((r) => {
      if (r?.original_filepath && r?.sample_id) {
        setActiveFilepath(r.original_filepath);
        setActiveSampleId(r.sample_id);
      }
      if (Array.isArray(r?.label_fields)) setAvailableLabelFields(r.label_fields);
    }).catch(() => {/* non-fatal */});
  }, [activeModalSlice, data?.sample_id]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Session restore when active sample/filepath changes ────────────────────
  const prevSettledIdRef   = useRef<string>("");
  const prevSettledPathRef = useRef<string>("");

  useEffect(() => { ensureSpinKeyframe(); }, []);

  // Elapsed-time counter
  useEffect(() => {
    if (isLoading) {
      editStartRef.current = Date.now();
      setElapsedSec(0);
      const id = setInterval(() => {
        setElapsedSec(Math.floor((Date.now() - editStartRef.current!) / 100) / 10);
      }, 100);
      return () => clearInterval(id);
    } else {
      editStartRef.current = null;
    }
  }, [isLoading]);

  // Pipeline status polling — fires every 1s while isLoading.
  useEffect(() => {
    if (!isLoading) {
      setPipelineStatus("");
      return;
    }
    const id = setInterval(() => {
      getPipelineStatus()
        .then((r) => {
          const s = r?.status ?? "";

          if (s === "done" && r.result) {
            clearInterval(id);
            const result = r.result as EditResult;
            const pending = pendingEditRef.current;
            if (pending) {
              const ep = result.effective_params;
              const newTurn: Turn = {
                images: result.images.map((img): TurnImage => ({
                  filepath: img.output_filepath,
                  imageUrl: img.image_data_url,
                })),
                prompt:              pending.prompt,
                model:               QWEN_MODEL_ID,
                timestamp:           Date.now(),
                generation_time:     result.generation_time,
                negative_prompt:     pending.negativePrompt || null,
                num_inference_steps: ep?.num_inference_steps ?? 40,
                true_cfg_scale:      ep?.true_cfg_scale      ?? 4.0,
                seed:                ep?.seed                ?? 51,
              };
              setTurns((prev) => [...prev, newTurn]);
              setSourceRef({ turnIdx: pending.expectedTurnIdx, imageIdx: 0 });
              pendingEditRef.current = null;
            }
            setPrompt("");
            setIsLoading(false);
            return;
          }

          if (s === "edit_error") {
            clearInterval(id);
            setErrorMsg(r.error ?? "Edit failed — check server logs.");
            pendingEditRef.current = null;
            setIsLoading(false);
            return;
          }

          if (s && s !== "ready") {
            setPipelineStatus(s);
          }
        })
        .catch(() => {/* ignore polling errors */});
    }, 1000);
    return () => clearInterval(id);
  }, [isLoading]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!activeSampleId || !activeFilepath) return;
    if (
      activeSampleId === prevSettledIdRef.current &&
      activeFilepath === prevSettledPathRef.current
    ) return;

    prevSettledIdRef.current   = activeSampleId;
    prevSettledPathRef.current = activeFilepath;

    const freshTurn = (): Turn => ({
      images: [{ filepath: activeFilepath, imageUrl: makeMediaUrl(activeFilepath) }],
      prompt: "",
      model: "",
      timestamp: 0,
    });

    const resetToFresh = () => {
      const t = freshTurn();
      setTurns([t]);
      setSourceRef({ turnIdx: 0, imageIdx: 0 });
      saveSession(activeSampleId, { turns: [t], sourceRef: { turnIdx: 0, imageIdx: 0 } });
    };

    setPrompt("");
    setNegativePrompt("");
    setNumSteps("");
    setTrueCfgScale("");
    setSeed("51");
    setNumImagesPerPrompt("1");
    setErrorMsg(null);
    setWarnMsg(null);
    setSaveErrorMsg(null);
    setSavingImages(new Set());
    setSavedImages(new Set());
    setDeletingImages(new Set());
    pendingEditRef.current = null;
    setIsLoading(false);

    const existing = loadSessionValidated(activeSampleId, activeFilepath);
    if (existing) {
      setTurns(existing.turns);
      setSourceRef(existing.sourceRef ?? { turnIdx: 0, imageIdx: 0 });
    } else {
      resetToFresh();
    }
  }, [activeSampleId, activeFilepath]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Session sync ────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!activeSampleId || !activeFilepath || turns.length === 0) return;
    if (turns[0].images[0]?.filepath !== activeFilepath) return;
    saveSession(activeSampleId, { turns, sourceRef });
  }, [turns, sourceRef, activeSampleId, activeFilepath]);

  // ── sourceRef safety net — reset if it points to a deleted image ────────────
  useEffect(() => {
    if (!turns.length) return;
    const turn = turns[sourceRef.turnIdx];
    if (!turn || !turn.images[sourceRef.imageIdx]) {
      setSourceRef({ turnIdx: 0, imageIdx: 0 });
    }
  }, [turns]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Scroll to bottom on new turns ───────────────────────────────────────────
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [turns.length, isLoading]);

  // ── Edit handler ────────────────────────────────────────────────────────────
  const handleEdit = useCallback(async () => {
    if (!prompt.trim() || isLoading || !activeFilepath) return;
    const sourceImg = turns[sourceRef.turnIdx]?.images[sourceRef.imageIdx];
    const sourceFilepath = sourceImg?.filepath ?? activeFilepath;

    // Pre-check: seed pipelineStatus with a meaningful message before the
    // operator starts so the user never sees a blank spinner.
    try {
      const preStatus = await getPipelineStatus();
      const s = preStatus?.status ?? "";
      if (s && s !== "ready" && s !== "done" && s !== "edit_error") {
        setPipelineStatus(s);
      }
    } catch {
      // Non-fatal — spinner falls back to "Editing image…".
    }

    pendingEditRef.current = {
      prompt: prompt.trim(),
      negativePrompt: negativePrompt.trim(),
      expectedTurnIdx: turns.length,
    };

    setIsLoading(true);
    setErrorMsg(null);
    setWarnMsg(null);

    const params: Record<string, any> = {
      prompt: prompt.trim(),
      input_filepath: sourceFilepath,
    };
    if (negativePrompt.trim())     params.negative_prompt      = negativePrompt.trim();
    if (numSteps !== "")           params.num_inference_steps  = Number(numSteps);
    if (trueCfgScale !== "")       params.true_cfg_scale       = Number(trueCfgScale);
    params.seed                  = seed !== "" ? Number(seed) : 51;
    if (numImagesPerPrompt !== "") params.num_images_per_prompt = Number(numImagesPerPrompt);

    runEdit(params as any).catch((e: any) => {
      setErrorMsg(e?.message ?? "Edit failed");
      pendingEditRef.current = null;
      setIsLoading(false);
    });
  }, [prompt, isLoading, activeFilepath, turns, sourceRef, negativePrompt, numSteps, trueCfgScale, seed, numImagesPerPrompt, runEdit, getPipelineStatus]);

  // ── Per-image save ───────────────────────────────────────────────────────────
  const handleSaveImage = useCallback((turnIdx: number, imageIdx: number) => {
    const imageKey = `${turnIdx}:${imageIdx}`;
    if (!activeSampleId || savingImages.has(imageKey)) return;
    const turn  = turns[turnIdx];
    const image = turn?.images[imageIdx];
    if (!turn || !image) return;

    const wasFlat = data?.dataset_is_grouped === false;
    setSavingImages((prev) => new Set(prev).add(imageKey));
    setSaveErrorMsg(null);

    saveExecutor
      .execute({
        sample_id:             activeSampleId,
        selected_label_fields: Array.from(selectedLabelFields),
        turns: [{
          filepath:             image.filepath,
          prompt:               turn.prompt,
          model:                turn.model,
          generation_time:      turn.generation_time,
          negative_prompt:      turn.negative_prompt  ?? null,
          num_inference_steps:  turn.num_inference_steps ?? null,
          true_cfg_scale:       turn.true_cfg_scale   ?? null,
          seed:                 turn.seed              ?? null,
        }],
        chat_history: turns.map((t) => ({
          filepath: t.images[0]?.filepath ?? "",
          prompt:   t.prompt,
          model:    t.model,
        })),
      })
      .then(() => {
        setSavingImages((prev) => { const n = new Set(prev); n.delete(imageKey); return n; });
        setSavedImages((prev) => new Set(prev).add(imageKey));
        if (wasFlat) {
          setWarnMsg(
            "Dataset converted to grouped format. " +
            "Close and reopen the sample modal for the display to refresh correctly."
          );
        }
      })
      .catch((e: any) => {
        setSaveErrorMsg(e?.message ?? "Save failed — check server logs.");
        setSavingImages((prev) => { const n = new Set(prev); n.delete(imageKey); return n; });
      });
  }, [activeSampleId, savingImages, turns, saveExecutor, selectedLabelFields, data?.dataset_is_grouped]);

  // ── Per-image delete ─────────────────────────────────────────────────────────
  const handleDeleteImage = useCallback((turnIdx: number, imageIdx: number) => {
    const imageKey = `${turnIdx}:${imageIdx}`;
    if (deletingImages.has(imageKey) || isLoading) return;
    const turn  = turns[turnIdx];
    const image = turn?.images[imageIdx];
    if (!turn || !image || turnIdx === 0) return;

    setDeletingImages((prev) => new Set(prev).add(imageKey));

    const updatedImages = turn.images.filter((_, ii) => ii !== imageIdx);
    const turnGone      = updatedImages.length === 0;

    deleteTurn({ filepath: image.filepath })
      .then(() => {
        setDeletingImages((prev) => { const n = new Set(prev); n.delete(imageKey); return n; });

        setTurns((prev) =>
          prev
            .map((t, ti) => ti !== turnIdx ? t : { ...t, images: updatedImages })
            .filter((t, ti) => ti === 0 || t.images.length > 0)
        );

        // Adjust sourceRef relative to the deletion
        setSourceRef((ref) => {
          // Deleted image was the source → reset
          if (ref.turnIdx === turnIdx && ref.imageIdx === imageIdx) {
            return { turnIdx: 0, imageIdx: 0 };
          }
          // Source is in the same turn but at a higher imageIdx → shift down
          if (ref.turnIdx === turnIdx && !turnGone && ref.imageIdx > imageIdx) {
            return { turnIdx: ref.turnIdx, imageIdx: ref.imageIdx - 1 };
          }
          // An earlier turn was fully deleted → shift source turnIdx down
          if (turnGone && ref.turnIdx > turnIdx) {
            return { turnIdx: ref.turnIdx - 1, imageIdx: ref.imageIdx };
          }
          return ref;
        });
      })
      .catch((e: any) => {
        setDeletingImages((prev) => { const n = new Set(prev); n.delete(imageKey); return n; });
        setSaveErrorMsg(`Delete failed: ${e?.message ?? e}`);
      });
  }, [deletingImages, isLoading, turns, deleteTurn]);

  // ── Early return ─────────────────────────────────────────────────────────────
  if (!activeFilepath) {
    return <div style={S.empty}>Open a sample to begin editing.</div>;
  }

  const canSend = !!prompt.trim() && !isLoading;
  const isSaving = saveExecutor.isLoading;

  return (
    <div style={S.root} onKeyDown={(e) => e.stopPropagation()}>

      {/* ── Top bar — Advanced toggle ── */}
      <div style={S.topBar}>
        <button
          style={S.advancedToggle}
          onClick={() => setShowAdvanced((v) => !v)}
        >
          {showAdvanced ? "▲ Advanced" : "▼ Advanced"}
        </button>
      </div>

      {/* ── Advanced settings ── */}
      {showAdvanced && (
        <div style={S.advancedPanel}>
          <div style={S.advancedRow}>
            <span style={S.advancedLabel}>Negative prompt</span>
            <input
              style={S.advancedInput}
              type="text"
              value={negativePrompt}
              onChange={(e) => setNegativePrompt(e.target.value)}
              placeholder="e.g. blurry, low quality"
            />
          </div>
          <div style={S.advancedRow}>
            <span style={S.advancedLabel}>Steps</span>
            <input
              style={S.advancedInput}
              type="number"
              min={1}
              max={150}
              value={numSteps}
              onChange={(e) => setNumSteps(e.target.value)}
              placeholder="default (40)"
            />
          </div>
          <div style={S.advancedRow}>
            <span style={S.advancedLabel}>True CFG scale</span>
            <input
              style={S.advancedInput}
              type="number"
              min={0}
              step={0.1}
              value={trueCfgScale}
              onChange={(e) => setTrueCfgScale(e.target.value)}
              placeholder="default (4.0)"
            />
          </div>
          <div style={S.advancedRow}>
            <span style={S.advancedLabel}>Seed</span>
            <input
              style={S.advancedInput}
              type="number"
              min={0}
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="51"
            />
          </div>
          <div style={S.advancedRow}>
            <span style={S.advancedLabel}>Images per prompt</span>
            <input
              style={S.advancedInput}
              type="number"
              min={1}
              value={numImagesPerPrompt}
              onChange={(e) => setNumImagesPerPrompt(e.target.value)}
              placeholder="1"
            />
          </div>

          {availableLabelFields.length > 0 && (
            <div style={{ marginTop: 6 }}>
              <div style={{ ...S.advancedRow, justifyContent: "space-between", marginBottom: 4 }}>
                <span style={{ ...S.advancedLabel, color: "#bbb", fontWeight: 600 }}>
                  Copy labels to saved slice
                </span>
                <div style={{ display: "flex", gap: 8 }}>
                  <button
                    style={{ ...S.advancedToggle, fontSize: 10 }}
                    onClick={() => setSelectedLabelFields(new Set(availableLabelFields))}
                  >all</button>
                  <button
                    style={{ ...S.advancedToggle, fontSize: 10 }}
                    onClick={() => setSelectedLabelFields(new Set())}
                  >none</button>
                </div>
              </div>
              {availableLabelFields.map((field) => (
                <div key={field} style={{ ...S.advancedRow, paddingLeft: 4, paddingTop: 2, paddingBottom: 2 }}>
                  <label style={{ display: "flex", alignItems: "center", gap: 7, cursor: "pointer", userSelect: "none" as const, width: "100%" }}>
                    <input
                      type="checkbox"
                      checked={selectedLabelFields.has(field)}
                      onChange={(e) => {
                        setSelectedLabelFields((prev) => {
                          const next = new Set(prev);
                          e.target.checked ? next.add(field) : next.delete(field);
                          return next;
                        });
                      }}
                      style={{ width: 13, height: 13, cursor: "pointer", accentColor: "#f5a623", flexShrink: 0 }}
                    />
                    <span style={{ ...S.advancedLabel, minWidth: 0, color: "#ccc", fontFamily: "monospace", fontSize: 11 }}>
                      {field}
                    </span>
                  </label>
                </div>
              ))}
            </div>
          )}

          <div style={{ ...S.advancedRow, justifyContent: "flex-end", marginTop: 6 }}>
            <button
              style={{ ...S.advancedToggle, fontSize: 11 }}
              onClick={() => {
                setNegativePrompt("");
                setNumSteps("");
                setTrueCfgScale("");
                setSeed("51");
                setNumImagesPerPrompt("1");
                setSelectedLabelFields(new Set());
              }}
            >
              ↺ reset
            </button>
          </div>
        </div>
      )}

      {/* ── Scrollable turn history ── */}
      <div ref={scrollRef} style={S.scroll}>
        {turns.map((turn, turnIdx) => (
          <div key={`${activeSampleId}-${turnIdx}`}>

            {/* Turn header — only for edited turns */}
            {turnIdx > 0 && (
              <div style={S.turnLabel}>
                <span>Turn {turnIdx}: &ldquo;{turn.prompt}&rdquo;</span>
                {turn.generation_time != null && (
                  <span style={S.turnTime}>{turn.generation_time}s</span>
                )}
              </div>
            )}

            {/* Per-image cards */}
            {turn.images.map((img, imageIdx) => {
              const isSource  = sourceRef.turnIdx === turnIdx && sourceRef.imageIdx === imageIdx;
              const imageKey  = `${turnIdx}:${imageIdx}`;
              const isDeleting = deletingImages.has(imageKey);

              return (
                <div
                  key={imageIdx}
                  style={{
                    ...S.imgWrap,
                    outline: isSource && turns.length > 1 ? "2px solid #f5a623" : "none",
                    borderRadius: 4,
                  }}
                >
                  {/* Image index badge for multi-image turns */}
                  {turn.images.length > 1 && (
                    <span style={S.imgIndexBadge}>
                      {imageIdx + 1} / {turn.images.length}
                    </span>
                  )}

                  <img
                    src={img.imageUrl}
                    alt={turnIdx === 0 ? "Original" : `Turn ${turnIdx} image ${imageIdx + 1}`}
                    style={S.img}
                  />

                  {turnIdx === 0 && <span style={S.badge}>Original</span>}

                  {/* Action buttons */}
                  {(turns.length > 1 || turnIdx > 0) && (
                    <div style={S.imgActions}>
                      {/* Set as source — available on all images when there are multiple turns */}
                      {turns.length > 1 && (
                        <button
                          style={{
                            ...S.imgActionBtn(false),
                            outline: isSource ? "1px solid #f5a623" : "none",
                          }}
                          onClick={() => setSourceRef({ turnIdx, imageIdx })}
                          title="Use as source for next edit"
                        >
                          <SourceIcon />
                        </button>
                      )}

                      {/* Trash + Save — only for edited turns */}
                      {turnIdx > 0 && (
                        <>
                          <button
                            style={S.imgActionBtn(isDeleting || isSaving || isLoading)}
                            onClick={() => handleDeleteImage(turnIdx, imageIdx)}
                            disabled={isDeleting || isSaving || isLoading}
                            title="Delete this edit"
                          >
                            <TrashIcon />
                          </button>
                          <button
                            style={S.imgActionBtn(savingImages.has(imageKey) || savedImages.has(imageKey) || isSaving)}
                            onClick={() => handleSaveImage(turnIdx, imageIdx)}
                            disabled={savingImages.has(imageKey) || savedImages.has(imageKey) || isSaving}
                            title={savedImages.has(imageKey) ? "Already saved as a group slice" : "Save as group slice"}
                          >
                            <SaveIcon />
                            <span style={{ fontSize: 11 }}>
                              {savingImages.has(imageKey) ? "Saving…" : savedImages.has(imageKey) ? "Saved ✓" : "Save"}
                            </span>
                          </button>
                        </>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ))}

        {isLoading && (
          <div style={{ ...S.loading, flexWrap: "wrap" as const, rowGap: 4 }}>
            <span style={S.spinner} />
            <span style={{ flex: 1, minWidth: 0 }}>
              {pipelineStatus || "Editing image…"}
            </span>
            <span style={{ fontVariantNumeric: "tabular-nums", opacity: 0.6, fontSize: 12, flexShrink: 0 }}>
              {elapsedSec.toFixed(1)}s
            </span>
          </div>
        )}

        {warnMsg     && <div style={S.warnBox}>⚠ {warnMsg}</div>}
        {errorMsg    && <div style={S.errorBox}>{errorMsg}</div>}
        {saveErrorMsg && <div style={S.errorBox}>{saveErrorMsg}</div>}
      </div>

      {/* ── Prompt input ── */}
      <div style={S.bottomBar}>
        <div style={S.inputRow}>
          <input
            type="text"
            style={S.input}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleEdit();
              }
            }}
            placeholder="Describe an edit… (Enter to send)"
            disabled={isLoading}
          />
          <button
            style={S.sendBtn(canSend)}
            onClick={handleEdit}
            disabled={!canSend}
            title="Edit image"
          >
            ✦
          </button>
        </div>
      </div>
    </div>
  );
};

export default QwenImageEditPanel;
