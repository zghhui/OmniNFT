import os
import os.path as osp
import math
import numpy as np
import torch
import torchaudio
from einops import rearrange
from torchvision.transforms import v2 as transforms_v2
from torio.io import StreamingMediaDecoder

from flow_grpo.audio_video_align.synchformer.synchformer import Synchformer, make_class_grid
from flow_grpo.audio_video_align.utils import pad_or_truncate, smart_pad


def av_desync_reward(device="cuda:0"):
    VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".mpeg", ".mpg", ".m4v")

    def _find_sidecar_audio(video_path):
        """固定同名 wav: xxx.mp4 -> xxx.wav"""
        if not isinstance(video_path, str):
            return None
        base, ext = os.path.splitext(video_path)
        if ext.lower() not in VIDEO_EXT:
            return None
        wav_path = base + ".wav"
        return wav_path if os.path.exists(wav_path) else None

    def _load_video_audio_as_tensors(
        video_path,
        audio_path,
        size=224,
        video_fps=25.0,
        audio_sr=16000,
        max_length_s=8.0,
    ):
        expected_video_length = int(video_fps * max_length_s)
        expected_audio_length = int(audio_sr * max_length_s)

        video_transform = transforms_v2.Compose([
            transforms_v2.Resize(size, interpolation=transforms_v2.InterpolationMode.BICUBIC),
            transforms_v2.CenterCrop(size),
            transforms_v2.ToImage(),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Load Video
        reader = StreamingMediaDecoder(video_path)
        reader.add_basic_video_stream(
            frames_per_chunk=expected_video_length,
            frame_rate=video_fps,
            format="rgb24",
        )
        reader.fill_buffer()
        data_chunk = reader.pop_chunks()
        video = data_chunk[0]

        video = video[:expected_video_length]
        video = smart_pad(video, expected_video_length - video.shape[0], dim=0)
        video = video_transform(video)  # (T, C, H, W)

        # Load Audio
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)  # mono

        if sample_rate != audio_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, audio_sr)

        audio = waveform[:expected_audio_length]
        audio = smart_pad(audio, expected_audio_length - audio.shape[0], dim=0)

        # Add batch dim
        video = video.unsqueeze(0).to(device)  # (1, T, C, H, W)
        audio = audio.unsqueeze(0).to(device)  # (1, Ta)
        return video, audio

    # ---- load model once ----
    ckpt_default = os.environ.get("SYNCHFORMER_CKPT", "checkpoints/synchformer_state_dict.pth")
    synchformer = Synchformer().to(device).eval()
    sd = torch.load(ckpt_default, map_location=device, weights_only=True)
    synchformer.load_state_dict(sd)

    sync_mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        win_length=400,
        hop_length=160,
        n_fft=1024,
        n_mels=128,
        wkwargs={"device": device},
    )
    mel_scale_fb = sync_mel.mel_scale.fb.to(device)
    sync_mel.mel_scale.register_buffer("fb", mel_scale_fb)

    sync_grid = make_class_grid(-2, 2, 21)

    @torch.no_grad()
    def _single_desync_score(video_path, max_length_s=8.0):
        final_audio_path = _find_sidecar_audio(video_path)
        if final_audio_path is None:
            raise ValueError(f"paired wav not found for video: {video_path}")

        video, audio = _load_video_audio_as_tensors(
            video_path=video_path,
            audio_path=final_audio_path,
            size=224,
            video_fps=25.0,
            audio_sr=16000,
            max_length_s=max_length_s,
        )

        # Step1: video feats
        b, t, c, h, w = video.shape
        assert b == 1 and c == 3 and h == 224 and w == 224

        v_seg, v_step = 16, 8
        nvs = (t - v_seg) // v_step + 1
        if nvs <= 0:
            return 2.0

        v_segments = [video[:, i * v_step:i * v_step + v_seg] for i in range(nvs)]
        vx = torch.stack(v_segments, dim=1)  # (1, S, T, C, H, W)
        vx = rearrange(vx, "b s t c h w -> (b s) 1 t c h w")
        vx = synchformer.extract_vfeats(vx)
        vx = rearrange(vx, "(b s) 1 t d -> b s t d", b=b)

        # Step2: audio feats
        _, ta = audio.shape
        a_seg, a_step = 10240, 5120
        nas = (ta - a_seg) // a_step + 1
        if nas <= 0:
            return 2.0

        a_segments = [audio[:, i * a_step:i * a_step + a_seg] for i in range(nas)]
        ax = torch.stack(a_segments, dim=1)

        ax = torch.log(sync_mel(ax) + 1e-6)
        ax = pad_or_truncate(ax, 66)
        ax = (ax - (-4.2677393)) / (2 * 4.5689974)
        ax = synchformer.extract_afeats(ax.unsqueeze(2))

        # Step3: compare
        frame_num = min(vx.shape[1], ax.shape[1])
        vx, ax = vx[:, :frame_num], ax[:, :frame_num]

        seg_size = 14
        seg_num = math.ceil(frame_num / seg_size)
        sync_scores = []

        for si in range(seg_num):
            fstart, fend = si * seg_size, min((si + 1) * seg_size, frame_num)
            vx_seg, ax_seg = vx[:, fstart:fend], ax[:, fstart:fend]
            flen = fend - fstart
            delta = seg_size - flen

            if delta > 0:
                if si == 0:
                    rep = math.ceil(delta / flen)
                    vpad = vx_seg.repeat(1, rep, *([1] * (vx_seg.dim() - 2)))[:, :delta]
                    apad = ax_seg.repeat(1, rep, *([1] * (ax_seg.dim() - 2)))[:, :delta]
                    vx_seg = torch.cat((vx_seg, vpad), dim=1)
                    ax_seg = torch.cat((ax_seg, apad), dim=1)
                else:
                    vx_seg = vx[:, -seg_size:]
                    ax_seg = ax[:, -seg_size:]

            logits = synchformer.compare_v_a(vx_seg, ax_seg)  # (1, 21)
            top_id = int(torch.argmax(logits, dim=-1).item())
            sync_scores.append(abs(sync_grid[top_id].item()))

        return float(np.mean(sync_scores)) if len(sync_scores) > 0 else 2.0

    def _fn(images, prompts, metadata):
        if not isinstance(images, (list, tuple)):
            images = [images]

        if metadata is None:
            metadata = [{} for _ in images]
        elif isinstance(metadata, dict):
            metadata = [metadata for _ in images]
        elif len(metadata) == 1 and len(images) > 1:
            metadata = [metadata[0] for _ in images]

        scores = []
        for item, md in zip(images, metadata):
            try:
                if not isinstance(item, str):
                    raise ValueError(f"video item must be path str, got {type(item)}")

                max_length_s = float(md.get("max_length_s", 8.0))
                d = _single_desync_score(video_path=item, max_length_s=max_length_s)

                # desync越小越好 -> reward越大越好
                s = 1.0 / (1.0 + d)
                if np.isnan(s) or np.isinf(s):
                    s = 0.0

            except Exception as e:
                print(f"[av_desync_reward] failed for item={item}, md={md}, err={repr(e)}")
                s = 0.0

            scores.append(float(s))

        return scores, {}

    return _fn