import { app } from '../../../scripts/app.js';

app.registerExtension({
  name: "chunker",

  async nodeCreated(node) {
    if (node.comfyClass === "Chunker") {
      // hide input index widget
      node.widgets[node.widgets.findIndex(({ name }) => name === "index")].hidden = true;
      //node.widgets[node.widgets.findIndex(({ name }) => name === "index")].height = 0;
      //node.widgets[node.widgets.findIndex(({ name }) => name === "index")].computedHeight = 0;

      // hide input index noodle
      //node.removeInput(node.inputs.findIndex(({ name }) => name === "index"));

      console.log(node);
    }
  },

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "Chunker") {
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
})
