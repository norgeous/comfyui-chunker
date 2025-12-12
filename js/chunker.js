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

const uploadFile = async (file, progressCallback) => {
  try {
    // Wrap file in formdata so it includes filename
    const body = new FormData();
    const i = file.webkitRelativePath.lastIndexOf('/');
    const subfolder = file.webkitRelativePath.slice(0, i+1);
    const new_file = new File([file], file.name, {
      type: file.type,
      lastModified: file.lastModified,
    });
    body.append("image", new_file);
    if (i > 0) body.append("subfolder", subfolder);
    const url = api.apiURL("/upload/image");
    const resp = await new Promise((resolve) => {
      let req = new XMLHttpRequest();
      req.upload.onprogress = (e) => progressCallback?.(e.loaded/e.total);
      req.onload = () => resolve(req);
      req.open('post', url, true);
      req.send(body);
    });
    if (resp.status !== 200) {
      alert(resp.status + " - " + resp.statusText);
    }
    return resp;
  } catch (error) {
    alert(error);
  }
};

const doUpload = async (file, node, pathWidget) => {
  const resp = await uploadFile(file, (p) => node.progress = p);
  node.progress = undefined;
  if (resp.status != 200) return false;
  const filename = JSON.parse(resp.responseText).name;
  pathWidget.options.values.push(filename);
  pathWidget.value = filename;
  pathWidget.callback?.(filename);
  return true;
};

const addUploadWidget = (that, nodeType, widgetName, buttonLabel) => {
  const node = that
  const pathWidget = that.widgets.find((w) => w.name === widgetName);
  const fileInput = document.createElement("input");
  chainCallback(that, "onRemoved", () => fileInput?.remove());
  Object.assign(fileInput, {
    type: "file",
    accept: ["video/mp4", "video/mov", "video/webm", "image/png", "image/jpeg", "audio/mp3"].join(','),
    style: "display: none",
    onchange: async () => {
      if (fileInput.files.length) {
        return await doUpload(fileInput.files[0], node, pathWidget)
      }
    },
  });
  that.onDragOver = (e) => !!e?.dataTransfer?.types?.includes?.('Files');
  that.onDragDrop = async function(e) {
    if (!e?.dataTransfer?.types?.includes?.('Files')) {
      return false;
    }
    const item = e.dataTransfer?.files?.[0];
    if (accept.includes(item?.type)) {
      return await doUpload(item, node, pathWidget);
    }
    return false;
  };
  document.body.append(fileInput);
  const uploadWidget = that.addWidget("button", buttonLabel, "image", () => {
    app.canvas.node_widget = null; //clear the active click event
    fileInput.click();
  });
  uploadWidget.options.serialize = false;
};

const fetchFirstFrame = (that, filename) => {
  const ext = filename.split(".").reverse()[0];
  if (['mp4'].includes(ext)) {
    fetch(`/api/chunker/get-first-frame?${new URLSearchParams({ filename })}`)
      .then(response => {
        if (!response.ok) console.error("chunker first-frame fetch failed");
        return response.json();
      }).then(data => {
        that.images = [data];
        const img = new Image();
        img.src = `/api/view?${new URLSearchParams(data)}`;
        that.imgs = [img];
      });
  }
  if (['jpg', 'jpeg', 'png'].includes(ext)) {
    const data = {
      type: 'input',
      subfolder: '',
      filename,
    };
    that.images = [data];
    const img = new Image();
    img.src = `/api/view?${new URLSearchParams(data)}`;
    that.imgs = [img];
  }
};

app.registerExtension({
  name: "comfyui-chunker",

  async setup(app) {
    app.api.addEventListener("execution_cached", (...data) => {
      // console.log("chunker execution_cached", data);
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

      "ChunkerLoad": () => {
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
          addUploadWidget(this, nodeType, "images", "choose file to upload");
        });
        // chainCallback(nodeType.prototype, "onConnectInput", function () {
        //   updateLabels(this);
        // });
        // chainCallback(nodeType.prototype, "onExecutionStart", function () {
        //   updateLabels(this);
        // });
        chainCallback(nodeType.prototype, "onExecuted", function (ui) {
          updateLabels(this, ui.values[0]);
        });
      },

      // "ChunkerMediaLoader": () => {
      //   chainCallback(nodeType.prototype, "onNodeCreated", function () {
      //     const imagesPath = this.widgets.find(({ name }) => name === "images");
      //     const masksPath = this.widgets.find(({ name }) => name === "masks");

      //     setTimeout(()=>{
      //       fetchFirstFrame(this, imagesPath.value);
      //     });

      //     imagesPath.callback = value => {
      //       if (imagesPath.previousValue !== value) imagesPath.previousValue = value;
      //       else return; // block execution if same value twice
      //       fetchFirstFrame(this, value);
      //     };

      //     // hide inputs
      //     //setTimeout(() => this.removeInput(this.inputs.findIndex(({ name }) => name === "store")));
      //     setTimeout(() => this.removeInput(this.inputs.findIndex(({ name }) => name === "image")));
      //     setTimeout(() => this.removeInput(this.inputs.findIndex(({ name }) => name === "image_paint")));
      //     //this.widgets.find(({ name }) => name === "image").computeSize = () => [0, -4];
      //     //this.widgets.find(({ name }) => name === "image_paint").computeSize = () => [0, -4];

      //     addUploadWidget(this, nodeType, "images", "choose images to upload");
      //     addUploadWidget(this, nodeType, "masks", "choose masks to upload");
      //   });
      //   chainCallback(nodeType.prototype, "onConnectInput", function () {
      //     updateLabels(this);
      //   });
      //   chainCallback(nodeType.prototype, "onExecutionStart", function () {
      //     updateLabels(this);
      //   });
      //   chainCallback(nodeType.prototype, "onExecuted", function (ui) {
      //     updateLabels(this, ui.values[0]);
      //   });
      // },

      "ChunkerDivide": () => {
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
          setTimeout(() => this.removeInput(this.inputs.findIndex(({ name }) => name === "store")));
        });
        chainCallback(nodeType.prototype, "onConnectInput", function () {
          updateLabels(this);
        });
        chainCallback(nodeType.prototype, "onExecutionStart", function () {
          updateLabels(this);
        });
        chainCallback(nodeType.prototype, "onExecuted", function (ui) {
          updateLabels(this, ui.values[0]);
          // const { timestamp2 } = this.store.get();
          // if (!timestamp2) this.store.set({ timestamp1: undefined, timestamp2: Date.now(), index: undefined, chunk_count: undefined });
        });
      },

      "ChunkerVACEToFirstLast": () => {
        chainCallback(nodeType.prototype, "onConnectInput", function () {
          updateLabels(this);
        });
        chainCallback(nodeType.prototype, "onExecutionStart", function () {
          updateLabels(this);
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
          element.insertAdjacentHTML("beforeEnd", `<video controls autoplay loop muted onloadstart="this.volume=0.5" style="display:block; width:100%; min-height:100px; height:calc(100% - ${statusHeight}px); background:var(--hdr-gradient);" />`);
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

            const { timestamp1, timestamp2, index = 0, chunk_count = 0 } = this.store.get();

            if (!timestamp2) {
              element.querySelector("#status").innerHTML = "Awaiting ChunkerDivide...";
              return;
            }

            const waitMillis = Date.now() - timestamp2;

            if (!timestamp1) {
              element.querySelector("#status").innerHTML = `Awaiting first chunk for ${humanMillis(waitMillis)}...`;
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
          updateLabels(this);
        });
        chainCallback(nodeType.prototype, "onExecutionStart", function () {
          updateLabels(this);
        });
        // chainCallback(nodeType.prototype, "onAfterExecuteNode", function (ui) {
        //   //console.log("onAfterExecuteNode");
        // });
        chainCallback(nodeType.prototype, "onExecuted", function (ui) {
          updateLabels(this, ui.values[0]);
          const { index, chunk_count, video_path } = ui.values[0];
          this.store.set({ timestamp1: this.store.get().timestamp2, timestamp2: Date.now(), index, chunk_count });
          const infoContainer = this.widgets.find(({ type }) => type === "ChunkInfoWidget").element;
          if (video_path) {
            const videoTag  = infoContainer.querySelector("video");
            videoTag.src = `/api/view?${new URLSearchParams(video_path).toString()}`;
            videoTag.volume = 0.5;
          }
        });
      },


      "ChunkerSave": () => {
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
          // create chunk info widget
          const element = document.createElement("div");
          element.insertAdjacentHTML("beforeEnd", `<style>:root {--hdr-gradient: linear-gradient(to top left in oklab, oklch(70% 0.5 340), oklch(90% 0.5 200));}</style>`); // todo
          element.insertAdjacentHTML("beforeEnd", `<video controls autoplay loop muted onloadstart="this.volume=0.5" style="display:block; width:100%; min-height:100px; height:calc(100%); background:var(--hdr-gradient);" />`);
          element.style.display = "flex";
          element.style.flexDirection = "column";
          element.style.gap = "2px";
          element.style.fontSize = "10px";
          element.style.textAlign = "center";
          element.style.color = "var(--descrip-text)";

          this.uuid = makeUUID();
          element.id = `chunk-info-${this.uuid}`;
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
        chainCallback(nodeType.prototype, "onExecuted", function (ui) {
          updateLabels(this, ui.values[0]);
          const { video_path } = ui.values[0];
          const infoContainer = this.widgets.find(({ type }) => type === "ChunkInfoWidget").element;
          if (video_path) {
            const videoTag  = infoContainer.querySelector("video");
            videoTag.src = `/api/view?${new URLSearchParams(video_path).toString()}`;
            videoTag.volume = 0.5;
          }
        });
      },

      "TensorDebug": () => {
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
          const element = document.createElement("pre");
          this.uuid = makeUUID();
          element.id = `tensor-info-${this.uuid}`;
          this.addDOMWidget(nodeData.name, "TensorInfo", element, {
            serialize: false,
            hideOnZoom: false,
            getHeight: () => 220,
          });
        });
        chainCallback(nodeType.prototype, "onExecuted", function (ui) {
          const data = ui.values[0];
          const infoContainer = this.widgets.find(({ type }) => type === "TensorInfo").element;
          infoContainer.innerText = JSON.stringify(data, null, 2);
        });
      },
    })[nodeData.name]?.()
  },
});
