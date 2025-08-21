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

const humanMillis = seconds => {
  if (seconds <= 0) return '-';
  const intervals = {
    years: 1000 * 60 * 60 * 24 * 7 * 52,
    weeks: 1000 * 60 * 60 * 24 * 7,
    days: 1000 * 60 * 60 * 24,
    hours: 1000 * 60 * 60,
    mins: 1000 * 60,
    secs: 1000,
    millis: 1,
  };
  const warn = ['years', 'weeks', 'days', 'hours'];
  const precision = { years: 2, weeks: 2, days: 2, hours: 2, mins: 1, secs: 0, millis: 0 };
  const [k, v] = Object.entries(intervals).find(([k, v]) => ~~(seconds / v));
  const value = +(seconds / v).toFixed(precision[k]);
  return `${value} ${value === 1 ? k.replace(/s$/, '') : k}${warn.includes(k) ? ' \u26A0\uFE0F' : ''}`;
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
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    ({
      "Chunker": () => {
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
          // try to hide index input
          this.widgets.find(({ name }) => name === "index").hidden = true;
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
          const statusHeight = 20;
          element.insertAdjacentHTML("beforeEnd", `<div id="status" style="height:${statusHeight}px; display:flex; flex-direction:column; justify-content:center;" />`);
          element.insertAdjacentHTML("beforeEnd", `<video controls autoplay loop style="display:block; width:100%; min-height:100px; height:calc(100% - ${statusHeight}px); background:black;" />`);
          element.style.display = "flex";
          element.style.flexDirection = "column";
          element.style.gap = "8px";
          element.style.fontSize = "10px";
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

            element.querySelector("#status").innerHTML = `<div>Next: ${humanMillis(nextChunkMillis)}, Final: ${humanMillis(finalChunkMillis)}</div><progress style="display:block; width:100%;" value="${chunks_completed}" max="${loop_count}" />`;
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

        // or onAfterExecuteNode
        chainCallback(nodeType.prototype, "onExecuted", function (ui) {
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
