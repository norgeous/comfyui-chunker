import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from av_load import av_load

plt.style.use('dark_background')


def plot_audio_waveform(waveform, sample_rate, title, output_path):
    num_channels = waveform.shape[0]
    num_samples = waveform.shape[1]
    time = np.arange(num_samples) / sample_rate

    width = min(40, max(12, num_samples / 5000))
    fig, axes = plt.subplots(num_channels, 1, figsize=(width, 4 * num_channels), squeeze=False)
    fig.suptitle(title, fontsize=14)

    for i in range(num_channels):
        axes[i, 0].plot(time, waveform[i].numpy(), linewidth=0.3)
        axes[i, 0].set_xlabel('Time (s)')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].set_title(f'Channel {i + 1}')
        axes[i, 0].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()


def main():
    mp4_files = [
        '/home/user/dev/save_load_combine/test-output/save_video_audio_stereo.mp4',
        '/home/user/dev/save_load_combine/test-output/save_video_audio_mono.mp4',
        '/home/user/dev/save_load_combine/test-output/save_audio_stereo.mp4',
        '/home/user/dev/save_load_combine/test-output/save_audio_mono.mp4',
        '/home/user/dev/save_load_combine/test-output/save_video_only.mp4',
        '/home/user/dev/save_load_combine/test-output/combine-3x30i-4o-equal_power.mp4',
        '/home/user/dev/save_load_combine/test-output/combine-3x30i-4o-newer_only.mp4',
        '/home/user/dev/save_load_combine/test-output/combine-3x30i-4o-ease_in_out.mp4',
        '/home/user/dev/save_load_combine/test-output/combine-3x30i-4o-linear.mp4',
        '/home/user/dev/save_load_combine/test-output/combine-3x30i-4o-older_only.mp4',
        '/home/user/dev/save_load_combine/test-output/combine-3x30i-0o-none.mp4',
        '/home/user/dev/save_load_combine/test-source/source3.mp4',
        '/home/user/dev/save_load_combine/test-source/source2.mp4',
        '/home/user/dev/save_load_combine/test-source/source1.mp4',
    ]

    output_dir = Path('/home/user/dev/save_load_combine/audio-graphs')
    output_dir.mkdir(exist_ok=True)

    for mp4_path in mp4_files:
        if not os.path.exists(mp4_path):
            print(f"Skipping {mp4_path} - file not found")
            continue

        print(f"Processing {mp4_path}")
        _, audio = av_load(mp4_path)

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
