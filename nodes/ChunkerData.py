from comfy_api.latest import io
from ..lib.utils_format import format_boolean, format_fps


class ChunkerData(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ChunkerData",
            display_name="\U0001F36B Data",
            category="chunker",
            description=(
                "Extract values from `chunker_data` as individual outputs, for use elsewhere in your workflow. "
            ),
            inputs=[
                io.Custom("CHUNKER_DATA").Input(
                    "chunker_data",
                    tooltip="Connect chunker_data from ChunkerRepeat node to here",
                ),
            ],
            outputs=[
                io.Custom("CHUNKER_DATA").Output(
                    "chunker_data",
                    tooltip="Pass through the chunker_data",
                ),
                io.Int.Output(
                    "chunk_length",
                    tooltip="Count of images in each chunk",
                ),
                io.Int.Output(
                    "overlap_length",
                    tooltip="Count of images to overlap between chunks",
                ),
                io.Int.Output(
                    "chunk_count",
                    tooltip="Count of chunks",
                ),
                io.Int.Output(
                    "total_length",
                    tooltip="Total length of output images",
                ),
                io.Int.Output(
                    "index",
                    tooltip="The current iteration index, ie; 0, 1, 2, ...",
                ),
                io.Float.Output(
                    "fps",
                    tooltip="FPS",
                ),
                io.Boolean.Output(
                    "is_i2v",
                    tooltip="True when images count > 0",
                ),
                io.Boolean.Output(
                    "is_first_chunk",
                    tooltip="True if this is the first chunk (index 0)",
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        self,
        chunker_data,
    ) -> io.NodeOutput:
        c = chunker_data["chunker_config"]
        is_first_chunk = chunker_data["index"] == 0
        is_i2v = chunker_data.get("is_i2v", False)

        ui_values = {
            "output_label_values": {
                "chunk_length": c["chunk_length"],
                "overlap_length": c["overlap_length"],
                "chunk_count": c["chunk_count"],
                "total_length": c["total_length"],
                "index": chunker_data["index"],
                "fps": format_fps(chunker_data["fps"]),
                "is_first_chunk": format_boolean(is_first_chunk),
                "is_i2v": format_boolean(is_i2v),
            },
        }

        return io.NodeOutput(
            chunker_data,
            c["chunk_length"],
            c["overlap_length"],
            c["chunk_count"],
            c["total_length"],
            chunker_data["index"],
            float(chunker_data["fps"]),
            is_i2v,
            is_first_chunk,
            ui={"values": [ui_values]},
        )
