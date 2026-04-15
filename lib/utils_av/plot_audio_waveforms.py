import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from av_load import av_load

plt.style.use('dark_background')


def plot_audio_waveform(waveform, sample_rate, title, output_path):
    waveform = waveform[0]
    num_channels = waveform.shape[0]
    num_samples = waveform.shape[1]

    width = min(40, max(12, num_samples / 5000))
    fig, axes = plt.subplots(num_channels, 1, figsize=(width, 4 * num_channels), squeeze=False)
    fig.suptitle(title, fontsize=14)

    for i in range(num_channels):
        axes[i, 0].plot(np.arange(num_samples), waveform[i].numpy(), linewidth=0.3)
        axes[i, 0].set_xlabel('Samples')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].set_title(f'Channel {i + 1}')
        # axes[i, 0].set_ylim(-32768, 32767)
        
        # Set x-axis ticks to multiples of sample_rate
        tick_locations = np.arange(0, num_samples + 1, sample_rate)
        axes[i, 0].set_xticks(tick_locations)
        axes[i, 0].set_xticklabels([int(loc) for loc in tick_locations])
        
        interval = sample_rate / 15
        for sample_pos in np.arange(0, num_samples, interval):
            axes[i, 0].axvline(x=sample_pos, color='red', linestyle='--', alpha=0.5)

    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    mp4_files = [
        'test-output/save_video_audio_stereo.mp4',
        'test-output/save_video_audio_mono.mp4',
        'test-output/save_audio_stereo.mp4',
        'test-output/save_audio_mono.mp4',
        'test-output/save_video_only.mp4',
        'test-output/combine-3x30i-4o-equal_power.mp4',
        'test-output/combine-3x30i-4o-newer_only.mp4',
        'test-output/combine-3x30i-4o-ease_in_out.mp4',
        'test-output/combine-3x30i-4o-linear.mp4',
        'test-output/combine-3x30i-4o-older_only.mp4',
        'test-output/combine-3x30i-0o-none.mp4',
        'test-source/source3.mp4',
        'test-source/source2.mp4',
        'test-source/source1.mp4',
    ]

    output_dir = Path('audio-graphs')
    output_dir.mkdir(exist_ok=True)

    for mp4_path in mp4_files:
        if not os.path.exists(mp4_path):
            print(f"Skipping {mp4_path} - file not found")
            continue

        print(f"Processing {mp4_path}")
        _, audio, _ = av_load(mp4_path)

        if audio is None:
            print(f"  No audio found in {mp4_path}")
            continue

        waveform = audio['waveform']
        sample_rate = audio['sample_rate']

        filename = Path(mp4_path).stem
        output_path = output_dir / f"{filename}_waveform.png"

        plot_audio_waveform(waveform, sample_rate, f"Audio Waveform: {filename}", output_path)
        print(f"  Saved graph to {output_path}")


if __name__ == "__main__":
    main()
