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

const setLabel = (puts, pname, label) => puts.find(({ name }) => name === pname).label = label;

app.registerExtension({
  name: "chunker",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    ({
      "Chunker": () => {
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
          // try to hide index input
          this.widgets.find(({ name }) => name === "index").hidden = true;

          // create progress widget
          const element = document.createElement("div");
          element.insertAdjacentHTML("afterbegin", '<progress style="display:block; width:100%; height:40px;" max="0" value="0" title="chunk 0 of 0" />')
          element.style.height = "40px";
          element.style.padding = "0 5px";
          this.uuid = makeUUID();
          element.id = `chunk-progress-${this.uuid}`;
          this.addDOMWidget(nodeData.name, "ChunkProgressWidget", element, {
            serialize: false,
            hideOnZoom: false,
          });
        });

        chainCallback(nodeType.prototype, "onExecuted", function (ui) {
          const { width, height, chunk_length, index, loop_count } = ui.values[0];

          // show width, height, chunk_length and index in labels
          setLabel(this.outputs, "width", `${width} width`);
          setLabel(this.outputs, "height", `${height} height`);
          setLabel(this.outputs, "chunk_length", `${chunk_length} chunk_length`);
          setLabel(this.outputs, "index", `${index} index`);

          // update the progress indicator
          const progressContainer = this.widgets.find(({ type }) => type === "ChunkProgressWidget");
          const progress = progressContainer.element.querySelector("progress");
          progress.value = index + 1;
          progress.max = loop_count;
          progress.title = `chunk ${index + 1} of ${loop_count}`;
        });
      },

      "ChunkerCombine": () => {
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
          // create eta widget
          const element = document.createElement("div");
          element.insertAdjacentHTML("afterbegin", 'ETA: <span id="eta">???</span>')
          element.style.padding = "0 5px";
          this.uuid = makeUUID();
          element.id = `chunk-eta-${this.uuid}`;
          this.addDOMWidget(nodeData.name, "ChunkEtaWidget", element, {
            serialize: false,
            hideOnZoom: false,
          });
        });

        chainCallback(nodeType.prototype, "onExecuted", function (ui) {
          // update images output count labels and eta
          const { image_count_in, image_count_out, eta } = ui.values[0];
          setLabel(this.inputs, "images", `images ${image_count_in}`);
          setLabel(this.outputs, "images", `${image_count_out} images`);
          const etaContainer = this.widgets.find(({ type }) => type === "ChunkEtaWidget");
          etaContainer.element.querySelector("#eta").innerHTML = eta;
        });
      },
    })[nodeData.name]?.()
  },
});
