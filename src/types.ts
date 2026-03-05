export interface TurnImage {
  filepath: string;
  imageUrl: string;
}

export interface Turn {
  /** Always at least one image. Turn 0 (original) always has exactly one. */
  images: TurnImage[];
  /** Empty string for turn 0 (the original). */
  prompt: string;
  /** Empty string for turn 0. */
  model: string;
  timestamp: number;
  /** API response time in seconds. Undefined for turn 0. */
  generation_time?: number;
  /** Null when no value was entered (turn 0 or field left blank). */
  negative_prompt?: string | null;
  num_inference_steps?: number | null;
  true_cfg_scale?: number | null;
  seed?: number | null;
}

export interface EditResult {
  /** Wall-clock seconds the pipeline call took. */
  generation_time: number;
  /**
   * One entry per generated image (num_images_per_prompt).
   * Each carries the temp filepath for saving and a JPEG data URL for
   * immediate in-panel display (bypasses FiftyOne's /media restriction).
   */
  images: { output_filepath: string; image_data_url: string }[];
  /**
   * The parameter values actually passed into the pipeline (defaults
   * resolved server-side). Stored on the Turn so saved samples always
   * reflect what was used even when the user left Advanced fields blank.
   */
  effective_params?: {
    true_cfg_scale: number;
    seed: number;
    num_inference_steps: number;
  };
}
