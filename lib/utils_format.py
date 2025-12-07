def format_audio(audio):
    if audio is None: return "0s"
    return f"{audio["waveform"].shape[2] / audio["sample_rate"]:.3f}s"

def format_fps(fps):
    if fps is None: return "0"
    return f"{fps:.2f}"
