import { registerComponent, PluginComponentType } from "@fiftyone/plugins";
import QwenImageEditPanel from "./QwenImageEditPanel";

/**
 * Register the React panel component.
 *
 * The ``name`` here must match the ``component`` kwarg passed to
 * ``types.View(component="QwenImageEditPanel", ...)`` in the Python
 * ``QwenImageEditPanel.render()`` method.
 */
// composite_view renders look up PluginComponentType.Component (type 3),
// not PluginComponentType.Panel (type 2).
registerComponent({
  name: "QwenImageEditPanel",
  component: QwenImageEditPanel,
  type: PluginComponentType.Component,
});
