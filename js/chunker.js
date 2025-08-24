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
  { p: 'years', s: 'year', ms: 1000 * 60 * 60 * 24 * 7 * 52, precision: 2, warn: true },
  { p: 'weeks', s: 'week', ms: 1000 * 60 * 60 * 24 * 7,      precision: 2, warn: true },
  { p: 'days',  s: 'day',  ms: 1000 * 60 * 60 * 24,          precision: 2, warn: true },
  { p: 'hours', s: 'hour', ms: 1000 * 60 * 60,               precision: 2, warn: true },
  { p: 'mins',  s: 'min',  ms: 1000 * 60,                    precision: 1 },
  { p: 'secs',  s: 'sec',  ms: 1000,                         precision: 0 },
];
const humanMillis = milliseconds => {
  if (milliseconds <= 0) return 'overdue';
  if (milliseconds < 1_000) return '< 1 sec';
  const { p, s, ms, precision, warn } = intervals.find(({ ms }) => ~~(milliseconds / ms));
  const value = +(milliseconds / ms).toFixed(precision);
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
    app.api.addEventListener("execution_error", (...data) => {
      console.log("execution_error", data);
    });
    app.api.addEventListener("execution_cached", (...data) => {
      console.log("execution_cached", data);
    });
    app.api.addEventListener("progress_state", (data) => {
      const nodes = Object.values(data.detail.nodes)
      console.log("progress_state", nodes, data.detail.nodes);
      const runningNode = nodes.findLast(({ state }) => state === "running");
      const runningNodeSiblings = nodes.filter(({ parent_node_id }) => runningNode.parent_node_id === parent_node_id);
      //console.log({ runningNodeSiblings });
      //const progress = / runningNodeSiblings.length;
    });
    app.api.addEventListener("progress_text", (...data) => {
      console.log("progress_text", data);
    });
    app.api.addEventListener("execution_interrupted", () => {
      document.querySelectorAll('#data_store').forEach(store => store.innerHTML = '{}');
    });
  },
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    ({
      "Chunker": () => {
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
          // try to hide index input
          // this.widgets.find(({ name }) => name === "index").hidden = true;
          this.widgets.find(({ name }) => name === "index").computeSize = [0, -4];

          // button
          this.addWidget("button", "Swap width / height", null, () => {
            console.log("click", this);
            const widthWidget = this.widgets.find(({ name }) => name === "width");
            const heightWidget = this.widgets.find(({ name }) => name === "height");
            const w = widthWidget.value;
            const h = heightWidget.value;
            widthWidget.value = h;
            heightWidget.value = w;
          });
        });

        chainCallback(nodeType.prototype, "onConnectInput", function () {
          updateLabels(this, { output_label_values: { width: undefined, height: undefined, chunk_length: undefined, index: undefined, loop_count: undefined }});
        });

        chainCallback(nodeType.prototype, "onExecuted", function (ui) {
          updateLabels(this, ui.values[0]);
        });
      },

      "ChunkerCombine": () => {
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
          //this.setSize([150, 100]);
          //this.resizable = false;

          // create chunk info widget
          const element = document.createElement("div");
          const statusHeight = 26;
          element.insertAdjacentHTML("beforeEnd", `<div id="status" style="height:${statusHeight}px; display:flex; flex-direction:column; justify-content:center;" />`);
          element.insertAdjacentHTML("beforeEnd", `<video controls autoplay loop style="display:block; width:100%; min-height:100px; height:calc(100% - ${statusHeight}px); background:black;" />`);
          element.style.display = "flex";
          element.style.flexDirection = "column";
          element.style.gap = "2px";
          element.style.fontSize = "8px";
          element.style.textAlign = "center";
          element.style.color = "var(--descrip-text)";

          this.store = jsonDivStore(element);
          setInterval(() => {
            const { timestamp1, timestamp2, index = 0, loop_count = 0 } = this.store.get();

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
            const chunks_remaining = loop_count - chunks_completed;
            const isDone = chunks_completed === loop_count;

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
              <progress style="display:block; width:100%;" value="${chunks_completed}" max="${loop_count}"></progress>
              <div>Showing up to ${chunks_completed} of ${loop_count}</div>
            `;
          }, 1_000);

          this.uuid = makeUUID();
          element.id = `chunk-info-${this.uuid}`;
          this.addDOMWidget(nodeData.name, "ChunkInfoWidget", element, {
            serialize: false,
            hideOnZoom: false,
            getHeight: () => 200,
          });
        });

        chainCallback(nodeType.prototype, "onConnectInput", function () {
          updateLabels(this, { input_label_values: { images: undefined }, output_label_values: { images: undefined }});
        });

        chainCallback(nodeType.prototype, "onExecutionStart", function () {
          updateLabels(this, { input_label_values: { images: undefined }, output_label_values: { images: undefined }});
          this.store.set({ timestamp1: undefined, timestamp2: Date.now(), index: undefined, loop_count: undefined });
        });

        chainCallback(nodeType.prototype, "onAfterExecuteNode", function (ui) {
          console.log("onAfterExecuteNode");
        });
        chainCallback(nodeType.prototype, "onExecuted", function (ui) {
          console.log("onExecuted");
          updateLabels(this, ui.values[0]);
          const { image_count, index, loop_count, video_path } = ui.values[0];
          this.store.set({ timestamp1: this.store.get().timestamp2, timestamp2: Date.now(), index, loop_count });

          const infoContainer = this.widgets.find(({ type }) => type === "ChunkInfoWidget").element;
          if (video_path) infoContainer.querySelector("video").src = `/api/view?${new URLSearchParams(video_path).toString()}`;
        });
      },
    })[nodeData.name]?.()
  },
});
