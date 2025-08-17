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

  async nodeCreated(node) {
    if (node.comfyClass === "Chunker") {
      // hide input index widget
      //node.widgets[node.widgets.findIndex(({ name }) => name === "index")].hidden = true;




      //node.widgets[node.widgets.findIndex(({ name }) => name === "index")].height = 0;
      //node.widgets[node.widgets.findIndex(({ name }) => name === "index")].computedHeight = 0;

      // hide input index noodle
      //node.removeInput(node.inputs.findIndex(({ name }) => name === "index"));

      console.log(node);
    }
  },

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "Chunker") {

      // progress widget
      chainCallback(nodeType.prototype, "onNodeCreated", function (got) {
        console.log({ got });
        this.widgets.find(({ name }) => name === "index").hidden = true;


        const element = document.createElement("progress");
        element.max = 100;
        element.value = 50;
        this.uuid = makeUUID();
        element.id = `chunk-progress-${this.uuid}`
        this.progress = this.addDOMWidget(nodeData.name, "ChunkProgressWidget", element, {
          // serialize: false,
          hideOnZoom: false,
        });
      });

      // show width, height, chunk_length and index in labels on execute
      const a = nodeType.prototype.onAfterExecuteNode;
      nodeType.prototype.onExecuted = function(ui) {
        const { width, height, chunk_length, index, loop_count } = ui.values[0];

        this.outputs.find(({ name }) => name === "width").label = `${width} width`;
        this.outputs.find(({ name }) => name === "height").label = `${height} height`;
        this.outputs.find(({ name }) => name === "chunk_length").label = `${chunk_length} chunk_length`;
        this.outputs.find(({ name }) => name === "index").label = `${index} of ${loop_count - 1} index`;

        const result = a?.apply(this, arguments);
        return result;
      }
    }
  },
});
