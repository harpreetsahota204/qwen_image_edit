import { useCallback } from "react";
import { usePanelEvent } from "@fiftyone/operators";

interface UpdateSliceResult {
  original_filepath: string;
  sample_id: string;
  label_fields: string[];
}

interface EditParams {
  prompt: string;
  input_filepath: string;
  negative_prompt?: string;
  num_inference_steps?: number;
  true_cfg_scale?: number;
  seed?: number;
  num_images_per_prompt?: number;
}

/**
 * Bridge to the Python ``QwenImageEditPanel`` panel methods:
 *   - ``run_edit``            – start inference in a background thread
 *   - ``update_slice``        – resolve filepath / sample-ID for a group slice
 *   - ``delete_turn``         – remove a temp edit file from disk
 *   - ``get_pipeline_status`` – poll the current load/inference status string
 *                               and deliver the final edit result
 */
export function usePanelClient(
  runEditUri: string,
  updateSliceUri: string,
  deleteTurnUri: string,
  getPipelineStatusUri: string,
) {
  const handleEvent = usePanelEvent();

  const runEdit = useCallback(
    (params: EditParams): Promise<{ status: string; error?: string }> =>
      new Promise((resolve, reject) => {
        handleEvent("run_edit", {
          operator: runEditUri,
          params,
          callback: (result: any) => {
            const r = result?.result as { status: string; error?: string } | undefined;
            if (r?.status === "edit_error") {
              reject(new Error(r.error ?? "Edit failed"));
            } else {
              resolve(r ?? { status: "started" });
            }
          },
        });
      }),
    [handleEvent, runEditUri]
  );

  const updateSlice = useCallback(
    (params: { slice: string }): Promise<UpdateSliceResult | null> =>
      new Promise((resolve, reject) => {
        handleEvent("update_slice", {
          operator: updateSliceUri,
          params,
          callback: (result: any) => {
            const err = result?.error ?? result?.result?.error;
            if (err) {
              reject(new Error(err));
            } else {
              resolve((result?.result as UpdateSliceResult) ?? null);
            }
          },
        });
      }),
    [handleEvent, updateSliceUri]
  );

  const deleteTurn = useCallback(
    (params: { filepath: string }): Promise<void> =>
      new Promise((resolve, reject) => {
        handleEvent("delete_turn", {
          operator: deleteTurnUri,
          params,
          callback: (result: any) => {
            const err = result?.error ?? result?.result?.error;
            err ? reject(new Error(err)) : resolve();
          },
        });
      }),
    [handleEvent, deleteTurnUri]
  );

  const getPipelineStatus = useCallback(
    (): Promise<{ status: string; result?: any; error?: string }> =>
      new Promise((resolve) => {
        handleEvent("get_pipeline_status", {
          operator: getPipelineStatusUri,
          params: {},
          callback: (result: any) => {
            resolve(
              (result?.result as { status: string; result?: any; error?: string })
                ?? { status: "" }
            );
          },
        });
      }),
    [handleEvent, getPipelineStatusUri]
  );

  return { runEdit, updateSlice, deleteTurn, getPipelineStatus };
}
