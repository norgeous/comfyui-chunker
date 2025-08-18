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

app.registerExtension({
  name: "chunker",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    ({
      "Chunker": () => {
        // progress widget
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
          this.widgets.find(({ name }) => name === "index").hidden = true;

          const element = document.createElement("div");
          element.insertAdjacentHTML("afterbegin", '<progress style="display:block; width:100%; height:40px;" max="0" value="0" title="chunk 0 of 0" />')
          element.style.height = "40px";
          element.style.padding = "0 5px";
          this.uuid = makeUUID();
          element.id = `chunk-progress-${this.uuid}`
          this.progress = this.addDOMWidget(nodeData.name, "ChunkProgressWidget", element, {
            // serialize: false,
            // hideOnZoom: false,
          });
        });

        // update labels
        chainCallback(nodeType.prototype, "onExecuted", function (ui) {
          const { width, height, chunk_length, index, loop_count } = ui.values[0];

          // show width, height, chunk_length and index in labels
          this.outputs.find(({ name }) => name === "width").label = `${width} width`;
          this.outputs.find(({ name }) => name === "height").label = `${height} height`;
          this.outputs.find(({ name }) => name === "chunk_length").label = `${chunk_length} chunk_length`;
          this.outputs.find(({ name }) => name === "index").label = `${index} index`;

          // update the progress indicator
          const progressContainer = this.widgets.find(({ type }) => type === "ChunkProgressWidget");
          const progress = progressContainer.element.querySelector("progress");
          progress.value = index + 1;
          progress.max = loop_count;
          progress.title = `chunk ${index + 1} of ${loop_count}`;
        });
      },

      "ChunkerCombine": () => {
        // update images output count label
        chainCallback(nodeType.prototype, "onExecuted", function (ui) {
          console.log("chunker combine ui", ui);
          const { image_count } = ui.values[0];
          this.outputs.find(({ name }) => name === "images").label = `${image_count} images`;
        });
      },
    })[nodeData.name]?.()
  },
});
