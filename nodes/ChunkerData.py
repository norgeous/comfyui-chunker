from comfy_api.latest import io
from ..lib.utils_format import format_fps


class ChunkerData(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ChunkerData",
            display_name="\U0001F36B Data",
            category="chunker",
            inputs=[
                io.Custom("CHUNKER_DATA").Input(
                    "chunker_data",
                    tooltip="Connect chunker_data from ChunkerRepeat node to here",
                ),
            ],
            outputs=[
                io.Int.Output(
                    "chunk_length",
                    tooltip="Count of images in each chunk",
                ),
                io.Int.Output(
                    "chunk_overlap",
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
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        self,
        chunker_data,
    ) -> io.NodeOutput:
        c = chunker_data["chunker_config"]

        ui_values = {
            "output_label_values": {
                "chunk_length": c["chunk_length"],
                "chunk_overlap": c["chunk_overlap"],
                "chunk_count": c["chunk_count"],
                "total_length": c["total_length"],
                "index": chunker_data["index"],
                "fps": format_fps(chunker_data["fps"]),
            },
        }

        return io.NodeOutput(
            c["chunk_length"],
            c["chunk_overlap"],
            c["chunk_count"],
            c["total_length"],
            chunker_data["index"],
            float(chunker_data["fps"]),
            ui={"values": [ui_values]},
        )
