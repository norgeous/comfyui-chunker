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
                io.Float.Output(
                    "fps",
                    tooltip="FPS",
                ),
                io.Int.Output(
                    "chunk_length",
                    tooltip="Count of images in each chunk",
                ),
                io.Int.Output(
                    "chunk_overlap",
                    tooltip="Count of images to overlap between chunks",
                ),
                io.Int.Output(
                    "total_length",
                    tooltip="Total length of output images",
                ),
                io.Int.Output(
                    "chunk_count",
                    tooltip="Count of chunks",
                ),
                io.Int.Output(
                    "index",
                    tooltip="The current iteration index, ie; 0, 1, 2, ...",
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
                "fps": format_fps(chunker_data["fps"]),
                "chunk_length": c["chunk_length"],
                "chunk_overlap": c["chunk_overlap"],
                "total_length": c["total_length"],
                "chunk_count": c["chunk_count"],
                "index": chunker_data["index"],
            },
        }

        return io.NodeOutput(
            float(chunker_data["fps"]),
            c["chunk_length"],
            c["chunk_overlap"],
            c["total_length"],
            c["chunk_count"],
            chunker_data["index"],
            ui={"values": [ui_values]},
        )
