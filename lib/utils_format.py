def format_images(images):
    return len(images) if images is not None else "0"

def format_masks(masks):
    return len(masks) if masks is not None else "0"

def format_audio(audio):
    if audio is None: return "0s"
    return f"{audio["waveform"].shape[2] / audio["sample_rate"]:.3f}s"

def format_latents(latents):
    if latents is None: return "0"
    if len(latents["samples"].shape) == 5:
        B, C, T, H, W = latents["samples"].shape
        if B == 1: return T
        return f"{B}×{T}"
    elif len(latents["samples"].shape) == 4:
        B, C, H, W = latents["samples"].shape
        return B

def format_fps(fps):
    if fps is None: return "0"
    return f"{fps:.2f}"

def format_milliseconds(ms):
    if ms == 0: return "0"
    divisors = [1, 1000, 60, 60, 24, 7, 52]
    units = ['ms', 's', 'm', 'h', 'd', 'w', 'y']
    results = []
    quotient = ms
    for i in range(1, len(divisors)):
        results.append(quotient % divisors[i])
        quotient //= divisors[i]
    results.append(quotient)
    rresults = list(reversed(results))
    runits = list(reversed(units))
    first = next(i for i,v in enumerate(rresults) if v > 0)
    last = len(results) - next(i for i,v in enumerate(results) if v > 0)
    out = [f"{rresults[i]}{runits[i]}" for i in range(first, last)]
    return ''.join(out[0:2])
