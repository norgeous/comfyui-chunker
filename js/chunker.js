import { app, ComfyApp } from '../../../scripts/app.js';
import { api } from '../../../scripts/api.js'

//from melmass
function makeUUID() {
  let dt = new Date().getTime()
  const uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = ((dt + Math.random() * 16) % 16) | 0
    dt = Math.floor(dt / 16)
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16)
  })
  return uuid
}

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

const statusHeight = 40;
document.body.insertAdjacentHTML("beforeEnd", `
<style>
:root {
  --hdr-gradient: linear-gradient(-45deg in oklab, oklch(30% 0.5 340), oklch(30% 0.5 200));
}
@keyframes animatepos {
  0%{background-position:0% 50%}
  50%{background-position:100% 50%}
  100%{background-position:0% 50%}
}
.chunker-status {
  height: ${statusHeight}px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 4px;
  font-size: 8px;
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
  gap: 1px;
  font-size: 3px;
}
.chunker-bar-section {
  flex: 1 1;
  padding: 1px;
  color: black;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  &.complete { background: #bada55; }
  &.current { background: linear-gradient(90deg, #bada55 0%, #bada55 50%, grey 50%, grey 100%); }
  &.pending { background: grey; }
}
.chunker-video {
  display: block;
  width: 100%;
  min-height: 100px;
  height: calc(100% - ${statusHeight}px);
  background: var(--hdr-gradient);
  background-size: 400% 400%;
  animation: animatepos 30s infinite;
}
</style>
`);

const formatMilliseconds = (ms, hideMs = false, pad = false) => {
  ms = Math.floor(ms)
  if (ms < 0) return "overdue";
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
  return value
}

const jsonDivStore = (element) => {
  element.insertAdjacentHTML("beforeEnd", '<pre id="data_store" style="font-size:8px; text-align:left; display:none;">{}</pre>');
  const storeElement = element.querySelector("#data_store");
  const get = () => JSON.parse(storeElement.innerHTML);
  const set = (data) => storeElement.innerHTML = JSON.stringify({ ...get(), ...data }, null, 2);
  return { get, set };
};

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

app.registerExtension({
  name: "chunker",

  async setup(app) {
    app.api.addEventListener("execution_start", () => {
      document.querySelectorAll('#data_store').forEach(store => store.innerHTML = '{}');
    });
    app.api.addEventListener("execution_interrupted", () => {
      document.querySelectorAll('#data_store').forEach(store => store.innerHTML = '{}');
    });
    app.api.addEventListener("execution_error", () => {
      document.querySelectorAll('#data_store').forEach(store => store.innerHTML = '{}');
    });
  },

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    ({
      "ChunkerDivide": () => {
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
          setTimeout(() => this.removeInput(this.inputs.findIndex(({ name }) => name === "store")));
        });
        chainCallback(nodeType.prototype, "onConnectInput", function () {
          updateLabels(this);
        });
        chainCallback(nodeType.prototype, "onExecutionStart", function () {
          // updateLabels(this);
        });
        chainCallback(nodeType.prototype, "onExecuted", function (ui) {
          updateLabels(this, ui.values[0]);
        });
      },

      "ChunkerVACEToFirstLast": () => {
        chainCallback(nodeType.prototype, "onConnectInput", function () {
          updateLabels(this);
        });
        chainCallback(nodeType.prototype, "onExecutionStart", function () {
          // updateLabels(this);
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
          this.uuid = makeUUID();
          element.id = `chunk-info-${this.uuid}`;
          element.insertAdjacentHTML("beforeEnd", `<div class="chunker-status" />`);
          element.insertAdjacentHTML("beforeEnd", `<video class="chunker-video" controls autoplay loop muted onloadstart="this.volume=0.5" />`);
          element.style.display = "flex";
          element.style.flexDirection = "column";
          element.style.gap = "2px";
          element.style.fontSize = "10px";
          element.style.textAlign = "center";
          element.style.color = "var(--descrip-text)";

          this.store = jsonDivStore(element);
          setInterval(() => {
            // fix the dodgy video width
            const vid = element.querySelector("video");
            vid.style.width = vid.style.width === "100%" ? "auto" : "100%";

            const { index, chunk_count, historical_deltas, predicted_deltas, ts } = this.store.get();

            const chunks_completed = index + 1;

            if (!historical_deltas) {
              element.querySelector(".chunker-status").innerHTML = `Awaiting data...`;
              return
            }

            const now = Date.now();
            const elapsedMillis = now - ts;
            const etaNextMillis = typeof predicted_deltas[0] === 'number' ? predicted_deltas[0] - elapsedMillis : 0;
            const etaFinalMillis = predicted_deltas ? predicted_deltas.reduce((acc, delta) => typeof delta === 'number' ? acc + delta : 0, 0) - elapsedMillis : 0;

            const etaNext = formatMilliseconds(etaNextMillis, true, true);
            const etaFinal = formatMilliseconds(etaFinalMillis, true, true);
            const dueDate = new Date(now + etaFinalMillis);
            const due = `${String(dueDate.getHours()).padStart(2, '0')}${Math.round(now / 1000) % 2 === 0 ? ':' : ' '}${String(dueDate.getMinutes()).padStart(2, '0')}`
            const warn = etaFinalMillis >= 1000 * 60 * 30; // 30 min

            const timings = `
              <div class="chunker-timings">
                <div class="chunker-timestamp">Next: ~${etaNext}</div>
                <div class="chunker-timestamp">Final: ~${etaFinal} @ ${due}${warn ? ' \u26A0\uFE0F' : ''}</div>
              </div>
            `;

            element.querySelector(".chunker-status").innerHTML = `
              ${predicted_deltas?.length ? timings : `Done in ${formatMilliseconds(historical_deltas.reduce((acc, delta) => typeof delta === 'number'? acc + delta : acc, 0))}`}
              <div class="chunker-bar">
                ${historical_deltas.map((delta, i) => {
                  const label = typeof delta === 'number' ? `${formatMilliseconds(delta)}` : delta;
                  return `<div class="chunker-bar-section complete" title="Chunk ${i + 1}\n${label}">${label}</div>`;
                }).join('\n')}
                ${predicted_deltas?.map((delta, i) => {
                  const label = typeof delta === 'number' ? `${formatMilliseconds(delta)}` : delta;
                  if (i === 0) {
                    const percent = typeof delta === 'number' ? Math.min(1, Math.max(0, (elapsedMillis) / delta)) * 100 : 50;
                    return `
                      <div
                        class="chunker-bar-section current"
                        style="background: linear-gradient(90deg, aqua 0%, aqua ${percent}%, grey ${percent}%, grey 100%);"
                        title="Chunk ${i + 1 + historical_deltas.length}\n~${label}"
                      >${label}</div>
                    `;
                  }
                  return `<div class="chunker-bar-section pending" title="Chunk ${i + 1 + historical_deltas.length}\n~${label}">${label}</div>`;
                }).join('\n')}
              </div>
              <div>Showing up to ${chunks_completed} of ${chunk_count}</div>
            `;
          }, 1_000);
          this.addDOMWidget(nodeData.name, "ChunkInfoWidget", element, {
            serialize: false,
            hideOnZoom: false,
            getHeight: () => 220,
          });
        });
        chainCallback(nodeType.prototype, "onConnectInput", function () {
          updateLabels(this);
        });
        chainCallback(nodeType.prototype, "onExecutionStart", function () {
          updateLabels(this);
        });
        chainCallback(nodeType.prototype, "onExecuted", async function (ui) {
          updateLabels(this, ui.values[0]);
          const now = Date.now();
          const {
            index,
            chunk_count,
            video_path,
            ts_chunk_starts,
            ts_chunk_ends,
          } = ui.values[0];
          const create_time = (await app.api.getQueue()).Running[0]?.create_time || (await app.api.getHistory())[0].create_time;
          const historical_deltas = ts_chunk_starts.reduce((acc, ts_chunk_start, index) => {
            const ts_chunk_end = ts_chunk_ends[index];
            const delta = ts_chunk_start < create_time ? "cached" : ts_chunk_end - ts_chunk_start;
            return [...acc, delta];
          }, []);
          const useful_historical_deltas = historical_deltas.filter(delta => typeof delta === 'number');
          const average = Math.round(useful_historical_deltas.reduce((acc, delta) => acc + delta, 0) / (useful_historical_deltas.length || 1)) || 'unknown';
          const predicted_deltas = Array.from({ length: chunk_count - historical_deltas.length }).fill(average);
          this.store.set({
            index,
            chunk_count,
            ts: now,
            historical_deltas,
            predicted_deltas,

            // debug vars
            create_time,
            ts_chunk_starts,
            ts_chunk_ends,
          });

          if (video_path) {
            const infoContainer = this.widgets.find(({ type }) => type === "ChunkInfoWidget").element;
            const videoTag  = infoContainer.querySelector("video");
            const videoParams = new URLSearchParams(video_path);
            videoParams.set("cache_buster", Math.random());
            videoTag.src = `/api/view?${videoParams.toString()}`;
          }
        });
      },

    })[nodeData.name]?.()
  },
});
