import { app } from '../../../scripts/app.js';

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

const intervals = [
  { p: 'years', s: 'year', ms: 1000 * 60 * 60 * 24 * 7 * 52, precision: 2 },
  { p: 'weeks', s: 'week', ms: 1000 * 60 * 60 * 24 * 7,      precision: 2 },
  { p: 'days',  s: 'day',  ms: 1000 * 60 * 60 * 24,          precision: 2 },
  { p: 'hours', s: 'hour', ms: 1000 * 60 * 60,               precision: 2 },
  { p: 'mins',  s: 'min',  ms: 1000 * 60,                    precision: 1 },
  { p: 'secs',  s: 'sec',  ms: 1000,                         precision: 0 },
];
const humanMillis = milliseconds => {
  if (milliseconds <= 0) return 'overdue';
  if (milliseconds < 1_000) return '< 1 sec';
  const { p, s, ms, precision } = intervals.find(({ ms }) => ~~(milliseconds / ms));
  const value = +(milliseconds / ms).toFixed(precision);
  const warn = milliseconds >= 1000 * 60 * 30; // 30 min
  return `${value} ${value === 1 ? s : p}${warn ? ' \u26A0\uFE0F' : ''}`;
};

const jsonDivStore = (element) => {
  element.insertAdjacentHTML("beforeEnd", '<pre id="data_store" style="display:none; font-size:8px; text-align:left;">{}</pre>');
  const storeElement = element.querySelector("#data_store");
  const get = () => JSON.parse(storeElement.innerHTML);
  const set = (data) => storeElement.innerHTML = JSON.stringify({ ...get(), ...data }, null, 2);
  return { get, set };
};

const updateLabels = (that, ui_values) => {
  Object.entries(ui_values.input_label_values || [])
    .forEach(([k, v]) => that.inputs.find(({ name }) => name === k).label = [k, v].filter(x => ![undefined, null].includes(x)).join(' '));
  Object.entries(ui_values.output_label_values || [])
    .forEach(([k, v]) => that.outputs.find(({ name }) => name === k).label = [v, k].filter(x => ![undefined, null].includes(x)).join(' '));
};

app.registerExtension({
  name: "comfyui-chunker",

  async setup(app) {
    app.api.addEventListener("execution_cached", (...data) => {
      //console.log("execution_cached", data);
    });
    app.api.addEventListener("execution_interrupted", () => {
      document.querySelectorAll('#data_store').forEach(store => store.innerHTML = '{}');
    });
    app.api.addEventListener("execution_error", (...data) => {
      //console.log("execution_error", data);
      document.querySelectorAll('#data_store').forEach(store => store.innerHTML = '{}');
    });
  },

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    ({
      "ChunkerConfig": () => {
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
          const mode = this.widgets.find(({ name }) => name === "mode");
          const chunk_length = this.widgets.find(({ name }) => name === "chunk_length");
          const chunk_overlap = this.widgets.find(({ name }) => name === "chunk_overlap");
          const total_length = this.widgets.find(({ name }) => name === "total_length");
          chainCallback(total_length, "callback", value => {
            console.log('total_length changed', { total_length, value });
          });
        });
        chainCallback(nodeType.prototype, "onExecuted", function (ui) {
          updateLabels(this, ui.values[0]);
        });
      },

      "ChunkerResequencer": () => {
        chainCallback(nodeType.prototype, "onConnectInput", function () {
          //updateLabels(this, { input_label_values: { images: undefined }, output_label_values: { images: undefined }});
        });

        chainCallback(nodeType.prototype, "onExecuted", function (ui) {
          updateLabels(this, ui.values[0]);
        });
      },

      "Chunker": () => {
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
          // hide store input
          setTimeout(() => this.removeInput(this.inputs.findIndex(({ name }) => name === "store")));
        });

        chainCallback(nodeType.prototype, "onConnectInput", function () {
          updateLabels(this, { output_label_values: { index: undefined }});
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
          const statusHeight = 32;
          element.insertAdjacentHTML("beforeEnd", `<div id="status" style="height:${statusHeight}px; display:flex; flex-direction:column; justify-content:center;" />`);
          element.insertAdjacentHTML("beforeEnd", `<video controls autoplay loop style="display:block; width:100%; min-height:100px; height:calc(100% - ${statusHeight}px); background:black;" />`);
          element.style.display = "flex";
          element.style.flexDirection = "column";
          element.style.gap = "2px";
          element.style.fontSize = "10px";
          element.style.textAlign = "center";
          element.style.color = "var(--descrip-text)";

          this.store = jsonDivStore(element);
          setInterval(() => {
            const { timestamp1, timestamp2, index = 0, chunk_count = 0 } = this.store.get();

            if (!timestamp2) {
              element.querySelector("#status").innerHTML = "";
              return;
            }

            const waitMillis = Date.now() - timestamp2;

            if (!timestamp1) {
              element.querySelector("#status").innerHTML = `Awaiting images for ${humanMillis(waitMillis)}...`;
              return;
            }

            const chunks_completed = index + 1;
            const chunks_remaining = chunk_count - chunks_completed;
            const isDone = chunks_completed === chunk_count;

            if (isDone) {
              element.querySelector("#status").innerHTML = "Done";
              return;
            }

            const lastChunkDelta = timestamp2 - timestamp1;
            const sinceLastTimestamp = Date.now() - timestamp2;
            const nextChunkMillis = lastChunkDelta - sinceLastTimestamp;
            const finalChunkMillis = (lastChunkDelta * chunks_remaining) - sinceLastTimestamp;

            element.querySelector("#status").innerHTML = `
              <div>Next: ${humanMillis(nextChunkMillis)}, Final: ${humanMillis(finalChunkMillis)}</div>
              <progress style="display:block; width:100%;" value="${chunks_completed}" max="${chunk_count}"></progress>
              <div>Showing up to ${chunks_completed} of ${chunk_count}</div>
            `;

            // attempt to fix the dodgy video width
            element.querySelector("video").style.width = Math.random() ? "100%" : "99.99%";
          }, 1_000);

          this.uuid = makeUUID();
          element.id = `chunk-info-${this.uuid}`;
          this.addDOMWidget(nodeData.name, "ChunkInfoWidget", element, {
            serialize: false,
            hideOnZoom: false,
            getHeight: () => 220,
          });
        });

        chainCallback(nodeType.prototype, "onConnectInput", function () {
          updateLabels(this, { input_label_values: { images: undefined }, output_label_values: { images: undefined }});
        });

        chainCallback(nodeType.prototype, "onExecutionStart", function () {
          updateLabels(this, { input_label_values: { images: undefined }, output_label_values: { images: undefined }});
          this.store.set({ timestamp1: undefined, timestamp2: Date.now(), index: undefined, chunk_count: undefined });
        });

        chainCallback(nodeType.prototype, "onAfterExecuteNode", function (ui) {
          //console.log("onAfterExecuteNode");
        });

        chainCallback(nodeType.prototype, "onExecuted", function (ui) {
          //console.log("onExecuted");
          updateLabels(this, ui.values[0]);
          const { image_count, index, chunk_count, video_path } = ui.values[0];
          this.store.set({ timestamp1: this.store.get().timestamp2, timestamp2: Date.now(), index, chunk_count });

          const infoContainer = this.widgets.find(({ type }) => type === "ChunkInfoWidget").element;
          if (video_path) infoContainer.querySelector("video").src = `/api/view?${new URLSearchParams(video_path).toString()}`;
        });
      },
    })[nodeData.name]?.()
  },
});
