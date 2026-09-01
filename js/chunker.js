const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

// from kjnodes
function chainCallback(object, property, callback) {
  if (object == undefined) {
      //This should not happen.
      console.error("Tried to add callback to non-existant object")
      return;
  }
  if (property in object) {
      const callback_orig = object[property]
      object[property] = function () {
          const r = callback_orig.apply(this, arguments);
          callback.apply(this, arguments);
          return r
      };
  } else {
      object[property] = callback;
  }
}

const statusHeight = 38;
document.body.insertAdjacentHTML("beforeEnd", `
<style>
@keyframes scrollLeftRight {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
@keyframes scrollRight {
  0% { background-position: 0% 50%; }
  100% { background-position: 100% 50%; }
}
.chunker-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-size: 8px;
  text-align: center;
  color: var(--descrip-text);
}
.chunker-status {
  height: ${statusHeight}px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 4px;
}
.chunker-timings {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
}
.chunker-timestamp {
  font-family: monospace;
}
.chunker-bar {
  display: flex;
  font-size: 3px;
  width: 100%;
}
.chunker-bar-section {
  flex: 1 1;
  padding: 1px 0;
  color: black;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  &.cached { background: #bada55; }
  &.complete { background: #57ab1e; }
  &.current { background: linear-gradient(90deg, aqua 0%, aqua 50%, grey 50%, grey 100%); }
  &.pending { background: grey; }
}
.chunker-video {
  display: block;
  width: 100%;
  min-height: 100px;
  height: calc(100% - ${statusHeight}px);
  &.hdr-gradient {
    background: linear-gradient(-45deg in oklab, oklch(30% 0.5 340), oklch(30% 0.5 200));
    background-size: 400% 400%;
    animation: scrollLeftRight 30s infinite;
  }
  &.checkerboard {
    background: repeating-conic-gradient(#333 0 25%, #111 0 50%) 50% / 32px 32px;
    animation: scrollRight 60s linear infinite;
  }
}

</style>
`);

const formatMilliseconds = (ms, hideMs = false, pad = false) => {
  if (!ms && ms !== 0) return 'unknown';
  const isNegative = ms < 0;
  ms = Math.floor(Math.abs(ms))
  if (ms === 0) return "0";
  const divisors = [1,    1000, 60,  60,  24,  7,   4,    13,  10,    10,  10];
  const units =    ['ms', 's',  'm', 'h', 'd', 'w', 'mo', 'y', 'dec', 'c', 'mi'];
  const results = [];
  let quotient = ms;
  for (let i = 1; i < divisors.length; i++) {
    results.push(quotient % divisors[i]);
    quotient = Math.floor(quotient / divisors[i]);
  }
  results.push(quotient);
  const rResults = [...results].reverse();
  const rUnits = [...units].reverse();
  const first = rResults.findIndex(v => v > 0);
  const last = results.length - results.findIndex(v => v > 0);
  const out = [];
  for (let i = first; i < last; i++) {
    const display = !pad ? rResults[i] : String(rResults[i]).padStart(2, '0');
    out.push(`${display}${rUnits[i]}`);
  }
  if (hideMs && out.length > 1) out.pop();
  const value = out.slice(0, 2).join('');
  return isNegative ? `${value} overdue` : value;
}



const updateLabels = (that, ui_values = {}) => {
  that.inputs.forEach(input => {
    const v = Object.entries(ui_values.input_label_values || {})?.find(([k]) => k === input.name)?.[1];
    input.label = [input.name, v].filter(x => x !== undefined).join(' ');
  });
  that.outputs.forEach(output => {
    const v = Object.entries(ui_values.output_label_values || {})?.find(([k]) => k === output.name)?.[1];
    output.label = [v, output.name].filter(x => x !== undefined).join(' ');
  });
};

const getBackgroundClass = (videoPath) => {
  const filename = new URLSearchParams(videoPath).get('filename') || '';
  return filename.endsWith('.webm') ? 'checkerboard' : 'hdr-gradient';
};

// Workaround for a frontend bug in migrateWidgetsValues (Comfy-Org/ComfyUI_frontend):
// during node configure it strips legacy pre-1.16 dummy widget values by
// positionally matching static input defs against widgets_values. fps has
// force_input=True, creating a phantom "forceInput" entry in the boolean array.
// When repeat_until is "input_length" (no DynamicCombo children), the widget
// count (4) coincides with the inputDefs count (4), so the migration strips
// index 0 — shifting all values by one and resetting repeat_until to its
// default. Dynamic children only exist in modern workflows, so the legacy
// migration can never legitimately apply: snapshot the saved values before
// configure, then re-apply all of them afterward.
const restoreDynamicComboValues = (nodeType) => {
  const origConfigure = nodeType.prototype.configure;
  if (typeof origConfigure !== 'function') return;
  nodeType.prototype.configure = function (info) {
    const saved = Array.isArray(info?.widgets_values)
      ? info.widgets_values.slice()
      : null;
    const result = origConfigure.apply(this, arguments);
    if (!saved || !this.widgets) return result;

    for (
      let i = 0;
      i < Math.min(saved.length, this.widgets.length);
      i++
    ) {
      const w = this.widgets[i];
      if (w && w.serialize !== false && w.value !== saved[i]) w.value = saved[i];
    }
    return result;
  };
};

app.registerExtension({
  name: "chunker",

  async setup() {
    api.addEventListener("execution_start", () => {
      document.querySelectorAll('#data_store').forEach(store => store.innerHTML = '{}');
    });
    api.addEventListener("execution_interrupted", () => {
      document.querySelectorAll('#data_store').forEach(store => store.innerHTML = '{}');
      app.graph._nodes
        .filter(n => n.type === "ChunkerCombine" && n.chunkerData)
        .forEach(n => {
          n.chunkerData.active = false;
        });
    });
    api.addEventListener("execution_error", () => {
      document.querySelectorAll('#data_store').forEach(store => store.innerHTML = '{}');
    });
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    ({
      "ChunkerRepeat": () => {
        restoreDynamicComboValues(nodeType);
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
          setTimeout(() => this.removeInput(this.inputs.findIndex(({ name }) => name === "store")));
        });
        chainCallback(nodeType.prototype, "onConnectInput", function () {
          updateLabels(this);
        });
        chainCallback(nodeType.prototype, "onExecuted", function (ui) {
          updateLabels(this, ui.values[0]);
        });
      },

      "ChunkerData": () => {
        chainCallback(nodeType.prototype, "onConnectInput", function () {
          updateLabels(this);
        });
        chainCallback(nodeType.prototype, "onExecuted", function (ui) {
          updateLabels(this, ui.values[0]);
        });
      },

      "ChunkerCombine": () => {
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
          // hide store input
          setTimeout(() => this.removeInput(this.inputs.findIndex(({ name }) => name === "store")));

          // create chunk info widget
          const element = document.createElement("div");
          element.className = "chunker-info"
          element.insertAdjacentHTML("beforeEnd", `<div class="chunker-status" />`);
          element.insertAdjacentHTML("beforeEnd", `<video class="chunker-video" controls autoplay loop muted />`);
          element.querySelector(".chunker-video").volume = 0.5;
          this.addDOMWidget(nodeData.name, "ChunkInfoWidget", element, {
            serialize: false,
            hideOnZoom: false,
            getHeight: () => 220,
          });
          this.chunkerData = {};
          const update = () => {
            const { active, lastCombineExecutionTs, bar } = this.chunkerData;
            const now = Date.now();
            const elapsedMillis = now - lastCombineExecutionTs;
            const currentDelta = bar?.find(({ type }) => type === 'current')?.delta;
            const currentPercent = Math.min(1, Math.max(0, (elapsedMillis / currentDelta) || 0.5)) * 100;
            const etaNextMillis = currentDelta ? currentDelta - elapsedMillis : undefined;
            const etaFinalMillis = bar?.reduce((acc, { type, delta }) => ['current', 'pending'].includes(type) && delta ? acc + delta : acc, 0) - elapsedMillis;
            const etaNext = `${etaNextMillis >= 0 ? '~' : ''}${formatMilliseconds(etaNextMillis, true, true)}`;
            const etaFinal = `${etaFinalMillis >= 0 ? '~' : ''}${formatMilliseconds(etaFinalMillis, true, true)}`;
            const dueDate = new Date(now + etaFinalMillis);
            const flashingSeparator = Math.round(now / 1000) % 2 === 0 ? ':' : ' ';
            const due = `${String(dueDate.getHours()).padStart(2, '0')}${flashingSeparator}${String(dueDate.getMinutes()).padStart(2, '0')}`;
            const warn = etaFinalMillis >= 1000 * 60 * 30; // 30 min
            const chunksCompleted = bar?.reduce((acc, { type }) => ['cached', 'complete'].includes(type) ? acc + 1 : acc, 0);
            const totalMillis = bar?.reduce((acc, { type, delta }) => ['cached', 'complete'].includes(type) && delta ? acc + delta : acc, 0);

            const statusText = !bar
              ? 'Awaiting execution...'
              : !active
                ? `Interrupted (Completed chunks took ${formatMilliseconds(totalMillis)})`
                : chunksCompleted === bar?.length
                  ? `Done in ${formatMilliseconds(totalMillis)}`
                  : undefined;

            element.querySelector(".chunker-status").innerHTML = `
              ${statusText ? `<div>${statusText}</div>` : ''}
              ${active !== false  && !statusText ? `
                <div class="chunker-timings">
                  <div class="chunker-timestamp">Next: ${etaNext}</div>
                  <div class="chunker-timestamp">Final: ${etaFinal} @ ${due}${warn ? ' \u26A0\uFE0F' : ''}</div>
                </div>
              ` : ''}
              ${bar ? `
                <div class="chunker-bar">
                  ${bar.map(({ type, delta }, i) => `
                    <div
                      class="chunker-bar-section ${type}"
                      ${type === 'current' && delta && active ? `style="background: linear-gradient(90deg, aqua 0%, aqua ${currentPercent}%, grey ${currentPercent}%, grey 100%);"` : ''}
                      title="Chunk ${i + 1}\n${formatMilliseconds(delta)}${type === 'cached' ? ' (cached)' : ''}"
                    >
                      ${formatMilliseconds(delta)}${type === 'cached' ? ' (cached)' : ''}
                    </div>`).join('\n')}
                </div>
                <div>Showing up to ${chunksCompleted} of ${bar.length}</div>
              ` : ''}
            `;
          };
          update();
          this.chunkerInterval = setInterval(update, 1_000);
        });
        chainCallback(nodeType.prototype, "onConnectInput", function () {
          updateLabels(this);
        });
        chainCallback(nodeType.prototype, "onExecuted", async function (ui) {
          updateLabels(this, ui.values[0]);
          const now = Date.now();
          const {
            bar,
            video_path,
          } = ui.values[0];
          this.chunkerData = {
            active: true,
            lastCombineExecutionTs: now,
            bar,
            video_path,
          };
          if (video_path) {
            const infoContainer = this.widgets.find(({ type }) => type === "ChunkInfoWidget").element;
            const videoTag  = infoContainer.querySelector("video");
            const videoParams = new URLSearchParams(video_path);
            videoParams.set("cache_buster", Math.random());
            videoTag.classList.remove('hdr-gradient', 'checkerboard');
            videoTag.classList.add(getBackgroundClass(video_path));
            videoTag.src = `/api/view?${videoParams.toString()}`;
          }
        });
        chainCallback(nodeType.prototype, "onSerialize", function (data) {
          if (this.chunkerData) {
            data.chunkerStore = this.chunkerData;
          }
        });
        chainCallback(nodeType.prototype, "onConfigure", function (data) {
          if (data.chunkerStore && Object.keys(data.chunkerStore).length > 1) {
            this.chunkerData = data.chunkerStore;
            if (data.chunkerStore.video_path) {
              const infoContainer = this.widgets?.find(({ type }) => type === "ChunkInfoWidget")?.element;
              if (infoContainer) {
                const videoTag = infoContainer.querySelector("video");
                if (videoTag) {
                  const videoParams = new URLSearchParams(data.chunkerStore.video_path);
                  videoParams.set("cache_buster", Math.random());
                  videoTag.classList.remove('hdr-gradient', 'checkerboard');
                  videoTag.classList.add(getBackgroundClass(data.chunkerStore.video_path));
                  videoTag.src = `/api/view?${videoParams.toString()}`;
                }
              }
            }
          }
        });
        chainCallback(nodeType.prototype, "onRemoved", function () {
          if (this.chunkerInterval) clearInterval(this.chunkerInterval);
        });
      },

    })[nodeData.name]?.()
  },
});
