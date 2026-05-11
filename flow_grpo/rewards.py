from PIL import Image
import io
import logging
import numpy as np
import torch
from collections import defaultdict
import os
import sys
import time

from flow_grpo.audio_video_align.av_desync import av_desync_reward

logger = logging.getLogger(__name__)

def hpsv3_score_video(device):
    import os
    import cv2
    import torch
    from PIL import Image
    import math
    from flow_grpo.remote_client import get_hpsv3_reward_score


    def _uniform_sample_all_frames_from_video(video_path_or_url):
        """Uniformly sample 5 frames from a video, return list of PIL.Image."""
        cap = cv2.VideoCapture(video_path_or_url)
        if not cap.isOpened():
            logger.warning(f"Cannot open video, skipping: {video_path_or_url}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        if total_frames is None or total_frames <= 0:
            grabbed = 0
            while grabbed < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
                grabbed += 1
            cap.release()
            return frames

        indices = [round(i * (total_frames - 1) / 4) for i in range(5)]
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

        cap.release()
        return frames

    def _is_video_path(x):
        if not isinstance(x, str):
            return False
        lower = x.lower()
        video_ext = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".mpeg", ".mpg", ".m4v")
        return lower.startswith(("http://", "https://", "rtsp://")) or lower.endswith(video_ext) or os.path.exists(x)

    def _single_image_score(img, prompt, reward_ip, port):
        """Score a single image via the reward API, return normalized score."""
        rewards = get_hpsv3_reward_score(
            prompts=[prompt],
            images=[img],
            ip=reward_ip,
            port=port
        )
        s = min(rewards[0][0].item(), 15)
        # s = rewards[0][0].item()
        # print(f"hps score: {s}")
        return s

    def _fn(images, prompts, metadata):
        hps_reward_ip = os.environ.get("HPSV3_REWARD_SERVER", "127.0.0.1")
        hps_reward_port = os.environ.get("HPSV3_REWARD_PORT", "8001")

        scores = []

        # Align input lengths
        prompts = [metadata_item["prompt_v"] for metadata_item in metadata]
        if not isinstance(images, (list, tuple)):
            images = [images]
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts] * len(images)
        elif len(prompts) == 1 and len(images) > 1:
            prompts = [prompts[0]] * len(images)

        for item, prompt in zip(images, prompts):
            # Video: sample 5 frames -> score each -> aggregate
            if _is_video_path(item):
                frames = _uniform_sample_all_frames_from_video(item)
                if len(frames) == 0:
                    scores.append(0.0)
                    continue

                frame_scores = []
                for f in frames:
                    frame_scores.append(_single_image_score(f, prompt, hps_reward_ip, hps_reward_port))
                # video_score = sum(frame_scores) / len(frame_scores)
                # Mean of top 30% frame scores
                sorted_scores = sorted(frame_scores, reverse=True)
                k = max(1, math.ceil(len(sorted_scores) * 0.3))
                video_score = sum(sorted_scores[:k]) / k
                scores.append(video_score)

            # Single image tensor
            elif isinstance(item, torch.Tensor):
                img = process_single_image(item)
                scores.append(_single_image_score(img, prompt, hps_reward_ip, hps_reward_port))

            # Other image inputs
            else:
                img = process_single_image(item)
                scores.append(_single_image_score(img, prompt, hps_reward_ip, hps_reward_port))

        return scores, {}

    return _fn

def videoalign_score(device):
    import os
    import requests
    import torch

    def _to_video_path(x):
        """Convert input to video path string; raise on unsupported types."""
        if isinstance(x, str):
            return x
        raise ValueError(f"videoalign_reward expects video path(str), got type={type(x)}")

    def _post_videoalign(video_paths, prompts, reward_ip, reward_port, fps=24, num_frames=121, timeout=300):
        url = f"http://{reward_ip}:{reward_port}/predict"
        payload = {
            "video_paths": video_paths,
            "prompts": prompts,
            "use_norm": True,
            
            # "num_frames": num_frames,
            # "max_pixels": 501760,
            "fps": 24.0
        }
        # print(payload)
        # r = requests.post(url, json=payload, timeout=timeout)
        # r.raise_for_status()
        # data = r.json()
        # print(data)

        total_attempts = 10 + 1
        last_err = None
        for i in range(total_attempts):
            try:
                r = requests.post(url, json=payload, timeout=timeout)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("status") == "success":
                        break
                last_err = f"http={r.status_code}, resp={r.text}"
            except Exception as e:
                last_err = str(e)
            if i < total_attempts - 1:
                time.sleep(3)

        # Expected format:
        # {"status":"success","rewards":[{"VQ":...,"MQ":...,"TA":...,"Overall":...}, ...]}
        rewards = data.get("rewards", [])
        scores = []
        for item in rewards:
            print(item)
            if isinstance(item, dict):
                s = (float(item.get("VQ", 0.0)) + float(item.get("TA", 0.0))) / 2
                scores.append(s)
                # if "Overall" in item:
                #     scores.append(float(item["Overall"]))
                # else:
                #     # fallback
                #     # print(item)
                #     s = (float(item.get("VQ", 0.0)) + float(item.get("MQ", 0.0)) + float(item.get("TA", 0.0))) / 3
                #     ## ensure video motion quality
                #     # vq, mq, ta = item.get("VQ", 0.0), item.get("MQ", 0.0), item.get("TA", 0.0)
                #     # vq = (vq + ta) / 2.0
                #     # s = mq * (0.4 + 0.6 * vq)
                #     scores.append(s)
            else:
                # In case the service returns a raw number
                scores.append(float(item))
        return scores

    def _fn(images, prompts, metadata):
        reward_ip = os.environ.get("VIDEOALIGN_REWARD_SERVER", "127.0.0.1")
        reward_port = os.environ.get("VIDEOALIGN_REWARD_PORT", "8000")

        # Align inputs
        if not isinstance(images, (list, tuple)):
            images = [images]

        prompts = [metadata_item["prompt_v"] for metadata_item in metadata]
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts] * len(images)
        elif len(prompts) == 1 and len(images) > 1:
            prompts = [prompts[0]] * len(images)

        if len(images) != len(prompts):
            raise ValueError(f"images/videos and prompts length mismatch: {len(images)} vs {len(prompts)}")

        # Convert to video paths
        video_paths = [_to_video_path(v) for v in images]

        # Call remote VideoAlign service
        scores = _post_videoalign(
            video_paths=video_paths,
            prompts=prompts,
            reward_ip=reward_ip,
            reward_port=reward_port,
            num_frames=121,
            timeout=300
        )

        # Return format: (scores, extra_info)
        return scores, {}

    return _fn

def audiobox_aesthetics_score(device):
    import os
    import subprocess
    import tempfile
    import numpy as np
    from audiobox_aesthetics.infer import initialize_predictor as initialize_audio_aes_predictor

    predictor = initialize_audio_aes_predictor(
        ckpt=os.environ.get("AUDIOBOX_CKPT", "checkpoints/audiobox-aesthetics/checkpoint.pt")
    )
    predictor.model = predictor.model.to(device)
    predictor.device = device

    VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".mpeg", ".mpg", ".m4v")
    AUDIO_EXT = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma")

    def _extract_wav_from_video(video_path, sr=16000):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_path = tmp.name
        tmp.close()

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", str(sr),
            "-f", "wav",
            wav_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return wav_path

    def _find_sidecar_wav(path_str):
        # Check for sidecar wav at the same path
        if not isinstance(path_str, str):
            return None
        base, ext = os.path.splitext(path_str)
        if ext.lower() in VIDEO_EXT:
            sidecar = base + ".wav"
            if os.path.exists(sidecar):
                return sidecar
        return None

    def _single_audio_score(item, sample_rate=16000):
        """
        Logic:
        1) Video path with sidecar wav -> use wav directly
        2) Video path without sidecar -> extract wav via ffmpeg
        3) Audio path -> use directly
        """
        tmp_wav = None
        try:
            if isinstance(item, str):
                lower = item.lower()
                _, ext = os.path.splitext(lower)

                if ext in VIDEO_EXT:
                    sidecar_wav = _find_sidecar_wav(item)
                    if sidecar_wav is not None:
                        wav_path = sidecar_wav
                    else:
                        wav_path = _extract_wav_from_video(item, sr=sample_rate)
                        tmp_wav = wav_path
                elif ext in AUDIO_EXT:
                    wav_path = item
                else:
                    # Non-standard extension: try sidecar wav first, then decode as video
                    sidecar_wav = _find_sidecar_wav(item)
                    if sidecar_wav is not None:
                        wav_path = sidecar_wav
                    else:
                        wav_path = _extract_wav_from_video(item, sr=sample_rate)
                        tmp_wav = wav_path
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")

            batch = [{"path": wav_path, "sample_rate": sample_rate}]
            outputs = predictor.forward(batch)  # list[dict], len=1
            out = outputs[0] if len(outputs) > 0 else {}
            ce = float(out.get('CE', 0.0)) if out is not None else 0.0
            cu = float(out.get('CU', 0.0)) if out is not None else 0.0
            pc = float(out.get('PC', 0.0)) if out is not None else 0.0
            pq = float(out.get('PQ', 0.0)) if out is not None else 0.0
            # print(outputs)
            # print(outputs)
            # raw_score = float(np.mean(list(outputs[0].values())))
            # raw_score = (ce + cu + pq + pc) / 40
            raw_score = (ce + cu + pq - pc) / 40
            return float(raw_score)
            # return float(raw_score/10.0)

        finally:
            if tmp_wav is not None and os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except Exception:
                    pass

    def _fn(images, prompts, metadata):
        scores = []

        if not isinstance(images, (list, tuple)):
            images = [images]
        
        prompts = [metadata_item["prompt_a"] for metadata_item in metadata]
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts] * len(images)
        elif len(prompts) == 1 and len(images) > 1:
            prompts = [prompts[0]] * len(images)

        sample_rate = 16000
        if isinstance(metadata, dict):
            sample_rate = int(metadata.get("sample_rate", 16000))

        for item, _prompt in zip(images, prompts):
            try:
                s = _single_audio_score(item, sample_rate=sample_rate)
            except Exception:
                s = 0.0
            scores.append(s)

        return scores, {}

    return _fn

def av_align_score(device):
    import os
    import numpy as np

    from flow_grpo.audio_video_align.av_align import (
        detect_audio_peaks,
        extract_frames,
        detect_video_peaks,
        calc_intersection_over_union,
    )

    VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".mpeg", ".mpg", ".m4v")
    AUDIO_EXT = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma")

    def _find_sidecar_audio(video_path):
        """Find sidecar audio: xxx.mp4 -> xxx.wav/mp3/flac..."""
        if not isinstance(video_path, str):
            return None
        base, ext = os.path.splitext(video_path)
        if ext.lower() not in VIDEO_EXT:
            return None
        for aext in AUDIO_EXT:
            cand = base + aext
            if os.path.exists(cand):
                return cand
        return None

    def _single_av_align_score(video_path, audio_path=None, size=None, max_length_s=None):
        # 1) Prefer sidecar audio
        sidecar_audio = _find_sidecar_audio(video_path)
        if sidecar_audio is not None:
            final_audio_path = sidecar_audio
        else:
            # 2) Fall back to metadata audio_path
            final_audio_path = audio_path

        if final_audio_path is None:
            raise ValueError(f"audio_path is missing for video: {video_path}")

        audio_peaks = detect_audio_peaks(final_audio_path, max_length_s=max_length_s)
        frames, fps = extract_frames(video_path, size, max_length_s=max_length_s)
        _, video_peaks = detect_video_peaks(frames, fps, use_tqdm=False)

        s = calc_intersection_over_union(audio_peaks, video_peaks, fps)
        if s is None or (isinstance(s, float) and np.isnan(s)):
            s = 0.0
        return float(s)

    def _fn(images, prompts, metadata):
        """
        Contract:
        - images: list of video paths (or single)
        - metadata: list of dicts, each may contain:
            - audio_path
            - size
            - max_length_s
        """
        scores = []

        if not isinstance(images, (list, tuple)):
            images = [images]

        # Align metadata
        if metadata is None:
            metadata = [{} for _ in range(len(images))]
        elif isinstance(metadata, dict):
            metadata = [metadata for _ in range(len(images))]
        elif len(metadata) == 1 and len(images) > 1:
            metadata = [metadata[0] for _ in range(len(images))]

        for item, md in zip(images, metadata):
            try:
                if not isinstance(item, str):
                    raise ValueError(f"video item must be path str, got {type(item)}")

                max_length_s = md.get("max_length_s", None)
                if max_length_s is not None:
                    max_length_s = float(max_length_s)

                s = _single_av_align_score(
                    video_path=item,
                    audio_path=None,
                    size=None,
                    max_length_s=max_length_s,
                )

                # Uncomment to clamp to [0,1]
                # s = max(0.0, min(1.0, s))

            except Exception as e:
                print(f"[av_align_score] failed for item={item}, md={md}, err={repr(e)}")
                s = 0.0

            scores.append(s)

        return scores, {}

    return _fn

def clap_score(device):
    import os
    import numpy as np
    import torch
    import torchaudio
    from transformers import ClapModel, AutoProcessor

    VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".mpeg", ".mpg", ".m4v")
    AUDIO_EXT = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma")

    CLAP_CKPT = os.environ.get("CLAP_CKPT", "checkpoints/clap-htsat-unfused")

    model = ClapModel.from_pretrained(CLAP_CKPT).eval().to(device=device)
    processor = AutoProcessor.from_pretrained(CLAP_CKPT)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def _find_sidecar_audio(video_path):
        """Find sidecar audio: xxx.mp4 -> xxx.wav/mp3/flac..."""
        if not isinstance(video_path, str):
            return None
        base, ext = os.path.splitext(video_path)
        if ext.lower() not in VIDEO_EXT:
            return None
        for aext in AUDIO_EXT:
            cand = base + aext
            if os.path.exists(cand):
                return cand
        return None

    def _load_audio_48k_mono(audio_path, max_length_s=None):
        wav, sr = torchaudio.load(audio_path)  # [C, T]
        # Convert to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        # Resample to 48kHz
        if sr != 48000:
            wav = torchaudio.functional.resample(wav, sr, 48000)
            sr = 48000
        # Truncate
        if max_length_s is not None:
            max_len = int(float(max_length_s) * sr)
            wav = wav[:, :max_len]
        return wav.squeeze(0).numpy()  # processor expects 1D numpy

    @torch.no_grad()
    def _single_clap_score(video_path, prompt, audio_path=None, max_length_s=None):
        # 1) Prefer sidecar audio
        sidecar_audio = _find_sidecar_audio(video_path)
        if sidecar_audio is not None:
            final_audio_path = sidecar_audio
        else:
            # 2) Fall back to metadata audio_path
            final_audio_path = audio_path

        if final_audio_path is None:
            raise ValueError(f"audio_path is missing for video: {video_path}")
        if not isinstance(prompt, str) or len(prompt.strip()) == 0:
            raise ValueError(f"prompt is empty for video: {video_path}")

        audio_arr = _load_audio_48k_mono(final_audio_path, max_length_s=max_length_s)

        inputs = processor(
            text=prompt,
            audios=audio_arr,
            return_tensors="pt",
            padding=True,
            truncation=True,
            sampling_rate=48000,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        s = cos(outputs.text_embeds, outputs.audio_embeds).mean().item()

        if np.isnan(s):
            s = 0.0
        return float(s)

    def _fn(images, prompts, metadata):
        """
        Contract:
        - images: list of video paths (or single)
        - prompts: list of text prompts (or single)
        - metadata: list/dict, may contain:
            - audio_path
            - max_length_s
        Returns:
        - scores: List[float]
        - extra: dict
        """
        scores = []
        prompts = [metadata_item["prompt_a"] for metadata_item in metadata]
        if not isinstance(images, (list, tuple)):
            images = [images]
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]

        # Align prompts length
        if len(prompts) == 1 and len(images) > 1:
            prompts = [prompts[0] for _ in range(len(images))]
        if len(prompts) != len(images):
            raise ValueError(f"len(prompts) != len(images): {len(prompts)} vs {len(images)}")

        # Align metadata
        if metadata is None:
            metadata = [{} for _ in range(len(images))]
        elif isinstance(metadata, dict):
            metadata = [metadata for _ in range(len(images))]
        elif len(metadata) == 1 and len(images) > 1:
            metadata = [metadata[0] for _ in range(len(images))]
        elif len(metadata) != len(images):
            raise ValueError(f"len(metadata) != len(images): {len(metadata)} vs {len(images)}")

        for item, prompt, md in zip(images, prompts, metadata):
            try:
                if not isinstance(item, str):
                    raise ValueError(f"video item must be path str, got {type(item)}")

                audio_path = md.get("audio_path", None)
                max_length_s = md.get("max_length_s", None)
                if max_length_s is not None:
                    max_length_s = float(max_length_s)

                s = _single_clap_score(
                    video_path=item,
                    prompt=prompt,
                    audio_path=audio_path,
                    max_length_s=max_length_s,
                )
                s = (s + 1.0) / 2.0
                s = max(0.0, min(1.0, s))
                
            except Exception as e:
                print(f"[clap_score] failed for item={item}, prompt={prompt}, md={md}, err={repr(e)}")
                s = 0.0

            scores.append(float(s))

        return scores, {}

    return _fn

def imagebind_sum_reward(device):
    """
    GRPO-style reward fn:
      input : images, prompts, metadata
      output: scores, extra_dict

    Per-sample scoring via ImageBind:
      score = sim(text_v, video) + sim(text_a, audio) + sim(audio, video)

    Contract:
    - images: video path(str) or list thereof
    - prompts: text or list (used as default v/a text)
    - metadata: dict or list[dict], optional fields:
        - audio_path: audio file path (falls back to sidecar audio)
        - prompt_v: video text (overrides prompts)
        - prompt_a: audio text (overrides prompts)
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ImageBind"))
    from imagebind import data as imagebind_data
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType

    AUDIO_EXT = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma")

    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval().to(device)
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def _find_sidecar_audio(video_path: str):
        if not isinstance(video_path, str):
            return None
        base, _ = os.path.splitext(video_path)
        for ext in AUDIO_EXT:
            cand = base + ext
            if os.path.exists(cand):
                return cand
        return None

    def _to_list(x, n=None, default=None):
        if isinstance(x, (list, tuple)):
            x = list(x)
        else:
            x = [x if x is not None else default]
        if n is not None:
            if len(x) == 1 and n > 1:
                x = x * n
            if len(x) != n:
                raise ValueError(f"Length mismatch: expect {n}, got {len(x)}")
        return x

    def _fn(images, prompts, metadata):
        # Align inputs
        videos = _to_list(images)
        n = len(videos)
        prompts_list = _to_list(prompts, n=n, default="")
        if metadata is None:
            metadata_list = [{} for _ in range(n)]
        elif isinstance(metadata, dict):
            metadata_list = [metadata for _ in range(n)]
        else:
            metadata_list = _to_list(metadata, n=n, default={})

        scores = []
        sim_tv_all, sim_ta_all, sim_av_all = [], [], []

        for vpath, p, md in zip(videos, prompts_list, metadata_list):
            try:
                if not isinstance(vpath, str):
                    raise ValueError(f"video path must be str, got {type(vpath)}")

                audio_path = md.get("audio_path", None)
                sidecar = _find_sidecar_audio(vpath)
                if sidecar is not None:
                    audio_path = sidecar
                if audio_path is None or not isinstance(audio_path, str):
                    raise ValueError(f"audio path missing for video: {vpath}")

                prompt_v = md.get("prompt_v", p)
                prompt_a = md.get("prompt_a", p)

                # Build ImageBind inputs
                inputs = {
                    ModalityType.VISION: imagebind_data.load_and_transform_video_data([vpath], device),
                    ModalityType.AUDIO: imagebind_data.load_and_transform_audio_data([audio_path], device),
                    # Two text inputs: video text + audio text
                    ModalityType.TEXT: imagebind_data.load_and_transform_text([prompt_v, prompt_a], device),
                }

                with torch.no_grad():
                    emb = model(inputs)

                # shape: [2, D] -> split into two text embeddings
                text_v, text_a = emb[ModalityType.TEXT][0:1], emb[ModalityType.TEXT][1:2]
                video_e = emb[ModalityType.VISION]   # [1, D]
                audio_e = emb[ModalityType.AUDIO]    # [1, D]

                sim_tv = cos(text_v, video_e).item()
                sim_ta = cos(text_a, audio_e).item()
                sim_av = cos(audio_e, video_e).item()

                score = float(sim_tv + sim_ta + sim_av) / 3
                scores.append(score)

                sim_tv_all.append(float(sim_tv))
                sim_ta_all.append(float(sim_ta))
                sim_av_all.append(float(sim_av))

            except Exception as e:
                print(f"[imagebind_sum_reward] failed: video={vpath}, md={md}, err={repr(e)}")
                scores.append(0.0)
                sim_tv_all.append(0.0)
                sim_ta_all.append(0.0)
                sim_av_all.append(0.0)

        return scores, {
            "sim_tv": sim_tv_all,
            "sim_ta": sim_ta_all,
            "sim_av": sim_av_all,
        }

    return _fn

def videoscore2_score(device):
    import os
    import requests
    import torch

    def _to_video_path(x):
        if isinstance(x, str):
            return x
        raise ValueError(f"videoscore2_score expects video path(str), got type={type(x)}")

    def _post_videoscore2(video_paths, prompts, reward_ip, reward_port, fps=None, timeout=600):
        url = f"http://{reward_ip}:{reward_port}/predict"
        payload = {
            "video_paths": video_paths,
            "prompts": prompts,
        }
        if fps is not None:
            payload["fps"] = fps

        total_attempts = 10 + 1
        last_err = None
        data = None
        for i in range(total_attempts):
            try:
                r = requests.post(url, json=payload, timeout=timeout)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("status") == "success":
                        break
                last_err = f"http={r.status_code}, resp={r.text}"
            except Exception as e:
                last_err = str(e)
            if i < total_attempts - 1:
                time.sleep(3)

        if data is None or data.get("status") != "success":
            raise RuntimeError(f"videoscore2 service failed after {total_attempts} attempts: {last_err}")

        rewards = data.get("rewards", [])
        scores = []
        for item in rewards:
            if isinstance(item, dict):
                if "error" in item:
                    logger.warning(f"videoscore2 error for {item.get('video_path')}: {item['error']}")
                    scores.append(0.0)
                    continue
                vq = item.get("visual_quality") or 0.0
                ta = item.get("text_to_video_alignment") or 0.0
                pc = item.get("physical_consistency") or 0.0
                scores.append((float(vq) + float(ta) + float(pc)) / 3.0)
            else:
                scores.append(float(item))
        return scores

    def _fn(images, prompts, metadata):
        reward_ip = os.environ.get("VS2_REWARD_SERVER", "127.0.0.1")
        reward_port = os.environ.get("VS2_REWARD_PORT", "8003")

        if not isinstance(images, (list, tuple)):
            images = [images]

        prompts = [metadata_item["prompt_v"] for metadata_item in metadata]
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts] * len(images)
        elif len(prompts) == 1 and len(images) > 1:
            prompts = [prompts[0]] * len(images)

        if len(images) != len(prompts):
            raise ValueError(f"images/videos and prompts length mismatch: {len(images)} vs {len(prompts)}")

        video_paths = [_to_video_path(v) for v in images]

        scores = _post_videoscore2(
            video_paths=video_paths,
            prompts=prompts,
            reward_ip=reward_ip,
            reward_port=reward_port,
            timeout=600,
        )

        return scores, {}

    return _fn

def multi_score(device, score_dict):
    score_functions = {
        "hpsv3_score_video": hpsv3_score_video,
        "audiobox_aesthetics_score": audiobox_aesthetics_score,
        "videoalign_score": videoalign_score,
        "av_align_score": av_align_score,
        "imagebind_sum_reward": imagebind_sum_reward,
        "av_desync_reward": av_desync_reward,
        "clap_score": clap_score,
        "videoscore2_score": videoscore2_score,
    }
    score_fns = {}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = (
            score_functions[score_name](device)
            if "device" in score_functions[score_name].__code__.co_varnames
            else score_functions[score_name]()
        )

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, only_strict=True):
        total_scores = []
        score_details = {}

        for score_name, weight in score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](
                    images, prompts, metadata, only_strict
                )
                score_details["accuracy"] = rewards
                score_details["strict_accuracy"] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f"{key}_strict_accuracy"] = value
                for key, value in group_rewards.items():
                    score_details[f"{key}_accuracy"] = value
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        score_details["avg"] = total_scores
        return score_details, {}

    return _fn


def main():

    prompts =["a trunk is on the road"]

    images = ["sample_video.mp4"]
    metadata = [{"prompt_v": prompts[0], "prompt_a": prompts[0]} ] # Example metadata
    score_dict = {"av_desync_reward": 1.0}
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()
