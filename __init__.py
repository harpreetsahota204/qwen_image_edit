"""FiftyOne Qwen Image Edit Panel plugin.

Architecture
------------
The plugin consists of two registered components:

``QwenImageEditPanel`` (foo.Panel)
    A hybrid panel that surfaces a React UI inside the sample modal.
    Python lifecycle hooks (``on_load``, ``on_change_current_sample``,
    ``on_change_group_slice``) push filepath / sample-ID state to the
    React component via ``ctx.panel.set_state()``.  Four callable panel
    methods are exposed to the frontend:

    * ``run_edit``            – validates parameters, starts inference in a
      daemon thread, and returns ``{"status": "started"}`` immediately.
    * ``update_slice``        – resolves the filepath/sample-ID for the
      currently selected group slice.
    * ``delete_turn``         – removes a temporary edit file from disk.
    * ``get_pipeline_status`` – returns the current pipeline load/inference
      status and delivers the final edit result when ready.

``SaveEditedImages`` (foo.Operator)
    An unlisted operator invoked from the React layer to persist one or
    more edited images as new ``edit_N`` group slices on the dataset.

Communication pattern
---------------------
FiftyOne reimports plugin modules on each panel-method call, which means
plain module-level globals (``x = None``) are reset to their initial values
on every invocation.  Two mechanisms survive this:

1. **Pipeline singleton** — stored as an attribute on a private fake
   module registered in ``sys.modules`` under the key
   ``"qwen_image_edit__persist"``.  FiftyOne never touches this entry,
   so the reference survives the fresh imports.

2. **Edit status & results** — written to a small JSON file on disk
   (``_STATUS_FILE``).  The background thread writes live step progress,
   then the final result (or error).  ``get_pipeline_status`` reads the
   file on each poll.  File I/O is inherently cross-process and
   cross-import safe.
"""

import base64
import copy
import io
import json
import os
import shutil
import sys
import threading
import time
import traceback

import bson
import torch
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from PIL import Image

import fiftyone as fo
import fiftyone.core.fields as fof
import fiftyone.operators as foo
import fiftyone.operators.types as types
from fiftyone import ViewField as F
from fiftyone.core.odm.database import get_db_conn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FP8_MODEL_REPO = "Qwen/Qwen-Image-Edit-2511"
FP8_TRANSFORMER_URL = (
    "https://huggingface.co/drbaph/Qwen-Image-Edit-2511-FP8/blob/main/"
    "qwen_image_edit_2511_fp8_e4m3fn.safetensors"
)

TEMP_DIR = os.path.join(os.path.expanduser("~"), ".fiftyone", "image_edits")

GROUP_FIELD = "group"
ORIGINAL_SLICE = "original"

# ---------------------------------------------------------------------------
# Pipeline singleton — survives module reimports
# ---------------------------------------------------------------------------
# FiftyOne does a *fresh* import of the plugin module on every panel-method
# call (not importlib.reload), so sys.modules[__name__] is a new object each
# time.  We stash the pipeline in a separate fake module registered under a
# private key that FiftyOne will never remove.

_PERSIST_KEY = "qwen_image_edit__persist"
if _PERSIST_KEY not in sys.modules:
    import types as _types
    _persist = _types.ModuleType(_PERSIST_KEY)
    _persist.pipeline = None
    sys.modules[_PERSIST_KEY] = _persist
else:
    _persist = sys.modules[_PERSIST_KEY]

_FP8_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "models--drbaph--Qwen-Image-Edit-2511-FP8",
)


def _is_transformer_cached() -> bool:
    """Return True when the FP8 transformer weights are already on disk."""
    return os.path.isdir(_FP8_CACHE_DIR)

# ---------------------------------------------------------------------------
# File-based status (survives module reimports and cross-process reads)
# ---------------------------------------------------------------------------

_STATUS_FILE = os.path.join(TEMP_DIR, ".edit_status.json")


def _write_status_file(data: dict) -> None:
    """Atomically write *data* as JSON to the status file."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    tmp = _STATUS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, _STATUS_FILE)


def _read_status_file() -> "dict | None":
    """Read the status file.  Returns ``None`` when no file exists."""
    try:
        with open(_STATUS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _clear_status_file() -> None:
    """Remove the status file (safe if already absent)."""
    try:
        os.remove(_STATUS_FILE)
    except FileNotFoundError:
        pass


def _set_status(msg: str) -> None:
    """Write a live status message for the active edit."""
    _write_status_file({"status": msg})


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------


def _get_qwen_diffusers_pipeline():
    """Lazy-load the Qwen FP8 pipeline once per server session.

    The loaded pipeline is stashed in ``_persist.pipeline`` (a fake
    module registered in ``sys.modules``) so it survives FiftyOne's
    per-call fresh imports of the plugin.
    """
    if _persist.pipeline is not None:
        return _persist.pipeline

    if _is_transformer_cached():
        _set_status("Loading transformer from cache…")
    else:
        _set_status("Downloading FP8 transformer (first time only)…")

    transformer = QwenImageTransformer2DModel.from_single_file(
        FP8_TRANSFORMER_URL,
        torch_dtype=torch.bfloat16,
        config=FP8_MODEL_REPO,
        subfolder="transformer",
    )

    _set_status("Loading pipeline to GPU…")
    transformer.enable_layerwise_casting(
        storage_dtype=torch.float8_e4m3fn,
        compute_dtype=torch.bfloat16,
    )
    _persist.pipeline = QwenImageEditPlusPipeline.from_pretrained(
        FP8_MODEL_REPO,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    return _persist.pipeline


def _make_step_callback(total_steps: int):
    """Return a diffusers ``callback_on_step_end`` that writes denoising
    progress to the status file for React polling."""
    def callback(pipeline, step, timestep, callback_kwargs):
        pct = int(round((step + 1) / total_steps * 100))
        _set_status(f"Step {step + 1}/{total_steps} ({pct}%)")
        return callback_kwargs
    return callback


# ---------------------------------------------------------------------------
# Core inference helper
# ---------------------------------------------------------------------------


def _run_edit(
    input_filepath: str,
    prompt: str,
    negative_prompt: str = None,
    num_inference_steps: int = 40,
    true_cfg_scale: float = 4.0,
    seed: int = 51,
    num_images_per_prompt: int = 1,
) -> tuple:
    """Run the Qwen FP8 pipeline and return all generated images.

    Returns
    -------
    tuple of (generation_time, images)
    """
    os.makedirs(TEMP_DIR, exist_ok=True)

    pipeline = _get_qwen_diffusers_pipeline()
    _set_status("Running inference…")

    input_image = Image.open(input_filepath).convert("RGB")

    t0 = time.time()
    with torch.inference_mode():
        output = pipeline(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt or " ",
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=torch.Generator("cuda").manual_seed(seed),
            num_images_per_prompt=num_images_per_prompt,
            callback_on_step_end=_make_step_callback(num_inference_steps),
        )
    generation_time = round(time.time() - t0, 2)

    torch.cuda.empty_cache()

    ts = int(time.time())
    base = os.path.splitext(os.path.basename(input_filepath))[0]

    images = []
    for idx, result_image in enumerate(output[0]):
        out_path = os.path.join(TEMP_DIR, f"{base}_{ts}_{idx}.png")
        result_image.save(out_path)

        display = result_image.copy()
        if max(display.width, display.height) > 1024:
            display.thumbnail((1024, 1024), Image.LANCZOS)
        disp_buf = io.BytesIO()
        display.save(disp_buf, format="JPEG", quality=85)
        image_data_url = (
            "data:image/jpeg;base64,"
            + base64.b64encode(disp_buf.getvalue()).decode()
        )

        images.append({
            "output_filepath": out_path,
            "image_data_url": image_data_url,
        })

    return generation_time, images


# ---------------------------------------------------------------------------
# Dataset / group helpers
# ---------------------------------------------------------------------------


def _copy_to_media_dir(src_filepath: str, original_filepath: str) -> str:
    """Copy an edited image into the same directory as the source sample.

    Placing the file alongside the original ensures it is reachable via the
    same media root that FiftyOne uses to serve the dataset.
    """
    dest_dir = os.path.dirname(original_filepath)
    dest_name = os.path.basename(src_filepath)
    dest_path = os.path.join(dest_dir, dest_name)
    shutil.copy2(src_filepath, dest_path)
    return dest_path


def _ensure_sample_in_group(dataset: fo.Dataset, sample: fo.Sample) -> str:
    """Ensure the dataset has a group field and that *sample* belongs to one.

    Handles three situations:

    1. **Flat dataset (no group field)** — adds a ``GROUP_FIELD`` group
       field, then bulk-assigns every existing sample to its own group
       (as the ``ORIGINAL_SLICE`` slice) via direct MongoDB writes.

    2. **Grouped dataset, sample not yet in a group** — creates a new
       ``fo.Group`` and assigns the sample to it as ``ORIGINAL_SLICE``.

    3. **Grouped dataset, sample already in a group** — returns the
       existing group ID unchanged.

    Returns
    -------
    str
        The group ID (as a hex string) that the sample belongs to.
    """
    gf = dataset.group_field

    if not gf:
        # ── Case 1: flat dataset ──────────────────────────────────────────
        dataset.add_group_field(GROUP_FIELD, default=ORIGINAL_SLICE)
        dataset.add_group_slice(ORIGINAL_SLICE, "image")
        gf = dataset.group_field

        db = get_db_conn()
        coll = db[dataset._sample_collection_name]

        target_group_id = None
        n = 0
        for doc in coll.find({gf: {"$exists": False}}):
            g = fo.Group()
            coll.update_one(
                {"_id": doc["_id"]},
                {"$set": {gf: {
                    "_id": bson.ObjectId(g.id),
                    "_cls": "Group",
                    "name": ORIGINAL_SLICE,
                }}},
            )
            if str(doc["_id"]) == sample.id:
                target_group_id = g.id
            n += 1

        dataset.reload()
        print(f"[image_edit] converted {n} samples to grouped ('{ORIGINAL_SLICE}' slice)")

        if target_group_id is None:
            raise RuntimeError(
                f"Sample {sample.id} not found in collection "
                f"'{dataset._sample_collection_name}'"
            )
        return target_group_id

    elif sample[gf] is None:
        # ── Case 2: grouped dataset, sample lacks a group entry ──────────
        group = fo.Group()
        sample[gf] = group.element(ORIGINAL_SLICE)
        sample.save()
        return group.id

    else:
        # ── Case 3: sample is already in a group ─────────────────────────
        return sample[gf].id


def _find_slice_sample(
    dataset: fo.Dataset,
    gf: str,
    group_id: str,
    slice_name: str,
) -> "fo.Sample | None":
    """Return the sample for *slice_name* within *group_id*, or ``None``."""
    return (
        dataset
        .select_group_slices(slice_name)
        .match(F(f"{gf}._id") == bson.ObjectId(group_id))
        .first()
    )


def _next_edit_slice_name(dataset: fo.Dataset, group_id: str) -> str:
    """Return the next unused ``edit_N`` slice name for the given group."""
    gf = dataset.group_field
    existing_names = set(
        dataset
        .select_group_slices()
        .match(F(f"{gf}._id") == bson.ObjectId(group_id))
        .values(f"{gf}.name")
    )
    idx = 1
    while f"edit_{idx}" in existing_names:
        idx += 1
    return f"edit_{idx}"


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

# Pre-declared so these fields appear in the sidebar immediately after
# reload_dataset(), even when their values are None on first save.
_OPTIONAL_EDIT_FIELDS: dict[str, type] = {
    "edit_negative_prompt":     fo.StringField,
    "edit_num_inference_steps": fo.IntField,
    "edit_true_cfg_scale":      fo.FloatField,
    "edit_seed":                fo.IntField,
}


def _ensure_edit_fields(dataset: fo.Dataset) -> None:
    """Declare optional edit-metadata fields if not already in the schema."""
    schema = dataset.get_field_schema()
    for name, ftype in _OPTIONAL_EDIT_FIELDS.items():
        if name not in schema:
            dataset.add_sample_field(name, ftype)


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


def _get_sample_label_fields(dataset: fo.Dataset, sample: fo.Sample) -> list[str]:
    """Return label field names that have a non-``None`` value on *sample*."""
    return [
        name
        for name, field in dataset.get_field_schema().items()
        if isinstance(field, fof.EmbeddedDocumentField)
        and issubclass(field.document_type, fo.core.labels.Label)
        and sample.get_field(name) is not None
    ]


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


class QwenImageEditPanel(foo.Panel):
    """Hybrid panel that provides chat-based image editing inside the modal."""

    @property
    def config(self):
        return foo.PanelConfig(
            name="qwen_image_edit_panel",
            label="Qwen Image Edit",
            surfaces="modal",
            help_markdown=(
                "Chat-based image editing powered by [drbaph/Qwen-Image-Edit-2511-FP8](https://huggingface.co/drbaph/Qwen-Image-Edit-2511-FP8). "
                "Type a prompt to edit the current image, then save your "
                "favourite results as new group slices on the dataset."
            ),
        )

    # ── Lifecycle hooks ──────────────────────────────────────────────────────

    def on_load(self, ctx):
        """Initialise panel state when the modal is first opened."""
        self._sync_sample(ctx)

    def on_change_current_sample(self, ctx):
        """Refresh state when the user navigates to a different sample."""
        self._sync_sample(ctx)

    def on_change_group_slice(self, ctx):
        """Refresh state when the active group slice changes."""
        self._sync_sample(ctx)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _sync_sample(self, ctx):
        """Push the current sample's filepath, ID, and label fields to React."""
        if not ctx.current_sample:
            return

        dataset = ctx.dataset
        sample = dataset[ctx.current_sample]
        gf = dataset.group_field

        filepath = sample.filepath
        sample_id = ctx.current_sample
        resolved_sample = sample

        if gf and ctx.group_slice:
            group_elem = sample[gf]
            if group_elem and group_elem.name != ctx.group_slice:
                try:
                    slice_sample = _find_slice_sample(dataset, gf, group_elem.id, ctx.group_slice)
                    if slice_sample is not None:
                        filepath = slice_sample.filepath
                        sample_id = slice_sample.id
                        resolved_sample = slice_sample
                        print(f"[image_edit] slice lookup OK -> sample_id={sample_id!r}")
                    else:
                        print(f"[image_edit] slice lookup returned None for slice={ctx.group_slice!r}")
                except Exception as exc:
                    print(f"[image_edit] slice lookup error: {exc}")

        ctx.panel.set_state("original_filepath", filepath)
        ctx.panel.set_state("sample_id", sample_id)
        ctx.panel.set_state("label_fields", _get_sample_label_fields(dataset, resolved_sample))
        ctx.panel.set_state("dataset_is_grouped", bool(dataset.group_field))

    # ── Panel methods (called from React via usePanelEvent) ──────────────────

    def run_edit(self, ctx):
        """Start an image edit in a background thread and return immediately.

        Seeds the status file with an initial message, then spawns a
        daemon thread.  The thread updates the status file with live
        step progress and writes the final result when done.

        Parameters (via ctx.params)
        ---------------------------
        prompt : str
        input_filepath : str
        negative_prompt : str, optional
        num_inference_steps : int, optional (default 40)
        true_cfg_scale : float, optional (default 4.0)
        seed : int, optional (default 51)
        num_images_per_prompt : int, optional (default 1)
        """
        prompt = ctx.params.get("prompt", "").strip()
        input_filepath = ctx.params.get("input_filepath", "")

        if not prompt:
            return {"status": "edit_error", "error": "Prompt cannot be empty."}
        if not input_filepath:
            return {"status": "edit_error", "error": "No input filepath provided."}

        negative_prompt = ctx.params.get("negative_prompt")
        num_inference_steps = int(ctx.params.get("num_inference_steps", 40))
        true_cfg_scale = float(ctx.params.get("true_cfg_scale", 4.0))
        seed = int(ctx.params.get("seed", 51))
        num_images_per_prompt = int(ctx.params.get("num_images_per_prompt", 1))

        if _persist.pipeline is None:
            _set_status(
                "Loading transformer from cache…"
                if _is_transformer_cached()
                else "Downloading FP8 transformer (first time only)…"
            )
        else:
            _set_status("Starting inference…")

        thread = threading.Thread(
            target=_run_edit_thread,
            kwargs=dict(
                input_filepath=input_filepath,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                seed=seed,
                num_images_per_prompt=num_images_per_prompt,
            ),
            daemon=True,
        )
        thread.start()
        return {"status": "started"}

    def get_pipeline_status(self, ctx):
        """Return the current pipeline load/inference status for React polling.

        Reads from the on-disk status file written by the background thread.

        Priority:
        1. ``"error"`` key   → edit failed.
        2. ``"done"`` key    → edit result ready.
        3. ``"status"`` key  → live progress (step counter, loading phase).
        4. No file           → idle / ready.
        """
        state = _read_status_file()
        if state is not None:
            if "error" in state:
                _clear_status_file()
                return {"status": "edit_error", "error": state["error"]}
            if "done" in state and "result" in state:
                _clear_status_file()
                return {"status": "done", "result": state["result"]}
            if "status" in state:
                return {"status": state["status"]}

        if _persist.pipeline is not None:
            return {"status": "ready"}
        if _is_transformer_cached():
            return {"status": "Idle — model not yet loaded"}
        return {"status": "Idle — model not downloaded"}

    def delete_turn(self, ctx):
        """Delete a temporary edited image file from disk.

        Only files inside ``TEMP_DIR`` may be deleted.

        Parameters (via ctx.params)
        ---------------------------
        filepath : str
            Absolute path to the temp file to remove.
        """
        filepath = ctx.params.get("filepath", "")
        if not filepath:
            return {"error": "No filepath provided."}

        abs_path = os.path.realpath(filepath)
        abs_temp = os.path.realpath(TEMP_DIR)
        if not abs_path.startswith(abs_temp + os.sep):
            return {"error": "Only temp edit files can be deleted."}

        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            return {"deleted": True}
        except Exception as exc:
            return {"error": str(exc)}

    def update_slice(self, ctx):
        """Resolve the filepath and sample-ID for a requested group slice.

        Called by React when the ``modalGroupSlice`` Recoil atom changes.
        Returns data directly (not via ``set_state``) because
        ``usePanelEvent`` synthesises a fake panel_id.

        Parameters (via ctx.params)
        ---------------------------
        slice : str
            The name of the slice the user switched to.
        """
        slice_name = ctx.params.get("slice", "")
        if not slice_name or not ctx.current_sample:
            return {}

        dataset = ctx.dataset
        sample = dataset[ctx.current_sample]
        gf = dataset.group_field

        if not gf:
            return {}

        group_elem = sample[gf]
        if not group_elem:
            return {}

        if group_elem.name == slice_name:
            return {
                "original_filepath": sample.filepath,
                "sample_id": ctx.current_sample,
                "label_fields": _get_sample_label_fields(dataset, sample),
            }

        try:
            slice_sample = _find_slice_sample(dataset, gf, group_elem.id, slice_name)
            if slice_sample is not None:
                return {
                    "original_filepath": slice_sample.filepath,
                    "sample_id": slice_sample.id,
                    "label_fields": _get_sample_label_fields(dataset, slice_sample),
                }
        except Exception as exc:
            print(f"[image_edit] update_slice error: {exc}")

        return {}

    def render(self, ctx):
        return types.Property(
            types.Object(),
            view=types.View(
                component="QwenImageEditPanel",
                composite_view=True,
                run_edit=self.run_edit,
                update_slice=self.update_slice,
                delete_turn=self.delete_turn,
                get_pipeline_status=self.get_pipeline_status,
            ),
        )


def _run_edit_thread(**kwargs) -> None:
    """Target for the daemon thread started by ``run_edit``.

    Runs ``_run_edit`` and writes the outcome to the status file so
    ``get_pipeline_status`` can deliver it to React on the next poll.
    """
    try:
        generation_time, images = _run_edit(**kwargs)
        _write_status_file({
            "done": True,
            "result": {
                "generation_time": generation_time,
                "images": images,
                "effective_params": {
                    "true_cfg_scale": kwargs["true_cfg_scale"],
                    "seed": kwargs["seed"],
                    "num_inference_steps": kwargs["num_inference_steps"],
                },
            },
        })
    except Exception as exc:
        _write_status_file({"error": str(exc)})


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


class SaveEditedImages(foo.Operator):
    """Persist one or more edited images as new ``edit_N`` group slices.

    This operator is unlisted and invoked programmatically from React via
    ``useOperatorExecutor``.  Each saved image is:

    * Copied from ``TEMP_DIR`` into the dataset's media directory.
    * Stored as a new ``fo.Sample`` in the same group as the source sample,
      on the next available ``edit_N`` slice.
    * Optionally populated with deep-copied label fields from the source.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="save_edited_images",
            label="Save Edited Images",
            unlisted=True,
        )

    def execute(self, ctx):
        """Persist the requested images and reload the dataset in the UI.

        Parameters (via ctx.params)
        ---------------------------
        sample_id : str
            ID of the source sample the edits were made from.
        turns : list[dict]
            Each dict represents one image to save, with keys ``filepath``,
            ``prompt``, ``model``, ``generation_time``, ``negative_prompt``,
            ``num_inference_steps``, ``true_cfg_scale``, and ``seed``.
        chat_history : list[dict]
            Full turn history stored as ``edit_history`` on each new sample.
        selected_label_fields : list[str], optional
            Label fields to deep-copy from the source sample onto each new
            edited sample.
        """
        try:
            sample_id = ctx.params["sample_id"]
            turns = ctx.params["turns"]
            chat_history = ctx.params["chat_history"]
            selected_label_fields: list[str] = ctx.params.get("selected_label_fields", [])

            dataset = ctx.dataset
            original_sample = dataset[sample_id]
            original_filepath = original_sample.filepath

            group_id = _ensure_sample_in_group(dataset, original_sample)
            gf = dataset.group_field

            _ensure_edit_fields(dataset)

            for turn in turns:
                saved_filepath = _copy_to_media_dir(turn["filepath"], original_filepath)
                slice_name = _next_edit_slice_name(dataset, group_id)

                edited_sample = fo.Sample(
                    filepath=saved_filepath,
                    **{gf: fo.Group(id=group_id).element(slice_name)},
                    parent_sample_id=sample_id,
                    edit_model=turn.get("model"),
                    edit_prompt=turn.get("prompt"),
                    edit_negative_prompt=turn.get("negative_prompt"),
                    edit_num_inference_steps=turn.get("num_inference_steps"),
                    edit_true_cfg_scale=turn.get("true_cfg_scale"),
                    edit_seed=turn.get("seed"),
                    edit_history=chat_history,
                    generation_time=turn.get("generation_time"),
                    tags=["edited"],
                )

                for field_name in selected_label_fields:
                    val = original_sample.get_field(field_name)
                    if val is not None:
                        edited_sample[field_name] = copy.deepcopy(val)

                dataset.add_sample(edited_sample)

                if slice_name not in dataset.group_slices:
                    dataset.add_group_slice(slice_name, "image")

            print(f"[image_edit] saved {len(turns)} edit slice(s) to group {group_id}")
            ctx.ops.reload_dataset()

        except Exception as e:
            print(f"[save_edited_images] ERROR: {e}")
            print(traceback.format_exc())
            raise


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(p):
    p.register(QwenImageEditPanel)
    p.register(SaveEditedImages)
