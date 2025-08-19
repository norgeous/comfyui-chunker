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

        chainCallback(nodeType.prototype, "onConnectionsChange", function () {
          updateLabels(this, {output_label_values:{width:undefined,height:undefined,chunk_length:undefined,index:undefined}});
        });

        chainCallback(nodeType.prototype, "onExecutionStart", function () {
          updateLabels(this, {output_label_values:{width:undefined,height:undefined,chunk_length:undefined,index:undefined}});
        });

        chainCallback(nodeType.prototype, "onExecuted", function (ui) {
          updateLabels(this, ui.values[0]);
        });
      },









      "ChunkerCombine": () => {
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
          // create chunk info widget
          const element = document.createElement("div");
          element.insertAdjacentHTML("beforeend", '<div id="data-store" style="display:none">{}</div>');
          element.insertAdjacentHTML("beforeend", '<div>Collected <span id="image_count">0</span> images so far</div>');
          element.insertAdjacentHTML("beforeend", '<div>Completed chunk <span id="current_chunk">0</span> of <span id="loop_count">0</span></div>');
          element.insertAdjacentHTML("beforeend", '<progress style="display:block; width:100%; height:40px;" max="0" value="0" />');
          element.insertAdjacentHTML("beforeend", '<div>ETA: <span id="eta">???</span></div>');
          element.style.padding = "0 10px";
          element.style.fontSize = "12px";

          this.uuid = makeUUID();
          element.id = `chunk-info-${this.uuid}`;
          this.addDOMWidget(nodeData.name, "ChunkInfoWidget", element, {
            serialize: false,
            hideOnZoom: false,
          });
        });

        chainCallback(nodeType.prototype, "onConnectionsChange", function () {
          updateLabels(this, {input_label_values:{images:undefined},output_label_values:{images:undefined}});
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
          updateLabels(this, {input_label_values:{images:undefined},output_label_values:{images:undefined}});
          const infoContainer = this.widgets.find(({ type }) => type === "ChunkInfoWidget").element;
          infoContainer.querySelector("#data").innerHTML = JSON.stringify({ startTimestamp: Date.now() });
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

          const { image_count, index, loop_count } = ui.values[0];

          // update chunk info widget
          const infoContainer = this.widgets.find(({ type }) => type === "ChunkInfoWidget").element;
          const { startTimestamp } = JSON.parse(infoContainer.querySelector("#data").innerHTML);
          infoContainer.querySelector("#eta").innerHTML = `s${startTimestamp}, n:${Date.now()}`;
          infoContainer.querySelector("#image_count").innerHTML = image_count;
          infoContainer.querySelector("#current_chunk").innerHTML = index + 1;
          infoContainer.querySelector("#loop_count").innerHTML = loop_count;
          const progress = infoContainer.querySelector("progress");
          progress.value = index + 1;
          progress.max = loop_count;
        });
      },
    })[nodeData.name]?.()
  },
});
