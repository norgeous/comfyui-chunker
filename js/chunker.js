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

const humanSeconds = seconds => {
  const intervals = {
    years: 1000 * 60 * 60 * 24 * 7 * 52,
    weeks: 1000 * 60 * 60 * 24 * 7,
    days: 1000 * 60 * 60 * 24,
    hours: 1000 * 60 * 60,
    minutes: 1000 * 60,
    seconds: 1000,
    milliseconds: 1,
  };

  const unit = Object.entries(intervals).find(([k, v]) => ~~(seconds / v)) || ['', 1];
  const value = +(seconds / unit[1]).toFixed(1);
  return `~${value} ${value === 1 ? unit[0].replace(/s$/, '') : unit[0]}`;
};

const jsonDivStore = (element) => {
  element.insertAdjacentHTML("afterstart", '<div id="data_store" style="display:block">{}</div>');
  const storeElement = element.querySelector("#data_store");
  const get = () => JSON.parse(storeElement.innerHTML);
  const set = (data) => storeElement.innerHTML = JSON.stringify({ ...get(), ...data });
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
          element.insertAdjacentHTML("beforeend", '<div>Got chunk <span id="current_chunk">0</span> of <span id="loop_count">0</span></div>');
          element.insertAdjacentHTML("beforeend", '<div>ETA: <span id="eta">???</span></div>');
          element.insertAdjacentHTML("beforeend", '<progress style="display:block; width:100%;" max="0" value="0" />');
          element.insertAdjacentHTML("beforeend", '<div><span id="image_count">0</span> images</div>');
          element.insertAdjacentHTML("beforeend", '<video controls autoplay loop style="width:100%; height:calc(100% - 54px); background: black;" />');
          //element.style.padding = "0 10px";
          element.style.fontSize = "12px";
          element.style.textAlign = "center";

          this.store = jsonDivStore(element);
          setTimeout(() => {
            const { startTimestamp, index = 0, loop_count = 0 } = this.store.get();

            if (!startTimestamp) {
              element.querySelector("#eta").innerHTML = "Awaiting first chunk...";
              return;
            }

            const elapsed = Date.now() - startTimestamp;
            const chunks_completed = index + 1;
            const chunks_remaining = loop_count - chunks_completed;
            const average_delta = elapsed / chunks_completed;
            const etaSeconds = chunks_remaining * average_delta;
            const eta = humanSeconds(etaSeconds);

            element.querySelector("#eta").innerHTML = eta;
          }, 1_000);

          this.uuid = makeUUID();
          element.id = `chunk-info-${this.uuid}`;
          this.addDOMWidget(nodeData.name, "ChunkInfoWidget", element, {
            serialize: false,
            hideOnZoom: false,
            getHeight: () => 70,
          });
        });

        chainCallback(nodeType.prototype, "onConnectInput", function () {
          updateLabels(this, { input_label_values: { images: undefined }, output_label_values: { images: undefined }});
          const infoContainer = this.widgets.find(({ type }) => type === "ChunkInfoWidget").element;
          infoContainer.querySelector("#eta").innerHTML = "???";
          infoContainer.querySelector("#image_count").innerHTML = "0";
          infoContainer.querySelector("#current_chunk").innerHTML = "0";
          infoContainer.querySelector("#loop_count").innerHTML = "0";
          const progress = infoContainer.querySelector("progress");
          progress.value = 0;
          progress.max = 0;
        });

        chainCallback(nodeType.prototype, "onExecutionStart", function () {
          updateLabels(this, { input_label_values: { images: undefined }, output_label_values: { images: undefined }});
          const infoContainer = this.widgets.find(({ type }) => type === "ChunkInfoWidget").element;
          this.store.set({ startTimestamp: Date.now() });
          infoContainer.querySelector("#eta").innerHTML = "???";
          infoContainer.querySelector("#image_count").innerHTML = "0";
          infoContainer.querySelector("#current_chunk").innerHTML = "0";
          infoContainer.querySelector("#loop_count").innerHTML = "0";
          const progress = infoContainer.querySelector("progress");
          progress.value = 0;
          progress.max = 0;
        });

        chainCallback(nodeType.prototype, "onExecuted", function (ui) {
          updateLabels(this, ui.values[0]);

          const { image_count, index, loop_count, video_path } = ui.values[0];

          this.store.set(ui.values[0]);

          // update chunk info widget
          const infoContainer = this.widgets.find(({ type }) => type === "ChunkInfoWidget").element;

          const { startTimestamp } = this.store.get();

          const elapsed = Date.now() - startTimestamp;
          const chunks_completed = index + 1;
          const chunks_remaining = loop_count - chunks_completed;
          const average_delta = elapsed / chunks_completed;
          const etaSeconds = chunks_remaining * average_delta;
          const eta = humanSeconds(etaSeconds);

          infoContainer.querySelector("#eta").innerHTML = eta;
          infoContainer.querySelector("#image_count").innerHTML = image_count;
          infoContainer.querySelector("#current_chunk").innerHTML = index + 1;
          infoContainer.querySelector("#loop_count").innerHTML = loop_count;

          if (video_path) infoContainer.querySelector("video").src = `/api/view?${new URLSearchParams(video_path).toString()}`;

          const progress = infoContainer.querySelector("progress");
          progress.value = chunks_completed;
          progress.max = loop_count;
        });
      },
    })[nodeData.name]?.()
  },
});
