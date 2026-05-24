import os
import logging
from typing import List, Optional

from pydub import AudioSegment
from pydub.effects import normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Timing constants (milliseconds)
CROSSFADE_MS = 100
INTER_SPEAKER_SILENCE_MS = 300
FADE_IN_MS = 200
FADE_OUT_MS = 200


def merge_audio_clips(
    clip_paths: List[str],
    final_output_path: str,
    headroom: float = 0.5,
    speakers: Optional[List[str]] = None,
):
    """
    Merges multiple audio clips into a single narration file with smooth
    transitions.  Adds per-clip fade-in / fade-out and inserts a short
    silence gap when the speaker changes.

    Args:
        clip_paths:        List of paths to audio clips (in order).
        final_output_path: Path for the final merged audio file.
        headroom:          dB of headroom below 0 dB (lower = louder).
        speakers:          Optional parallel list of speaker names (same
                           length as *clip_paths*).  Used to detect speaker
                           changes and insert silence gaps.
    """
    if not clip_paths:
        logger.warning("No audio clips to merge.")
        return

    logger.info(f"Merging {len(clip_paths)} clips into {final_output_path}...")

    try:
        first_clip = AudioSegment.from_file(clip_paths[0])
        # Apply fades to the first clip
        first_clip = first_clip.fade_in(min(FADE_IN_MS, len(first_clip)))
        first_clip = first_clip.fade_out(min(FADE_OUT_MS, len(first_clip)))
        combined = first_clip

        silence_gap = AudioSegment.silent(duration=INTER_SPEAKER_SILENCE_MS)

        for i, path in enumerate(clip_paths[1:], start=1):
            next_clip = AudioSegment.from_file(path)
            # Per-clip fades
            next_clip = next_clip.fade_in(min(FADE_IN_MS, len(next_clip)))
            next_clip = next_clip.fade_out(min(FADE_OUT_MS, len(next_clip)))

            # Determine whether the speaker changed
            speaker_changed = False
            if speakers and i < len(speakers):
                speaker_changed = speakers[i] != speakers[i - 1]

            if speaker_changed:
                # Insert silence gap then append without crossfade
                combined = combined + silence_gap + next_clip
            else:
                actual_crossfade = min(CROSSFADE_MS, len(combined), len(next_clip))
                combined = combined.append(next_clip, crossfade=actual_crossfade)

        # Normalize volume
        combined = normalize(combined, headroom=headroom)

        ext = os.path.splitext(final_output_path)[1].lower().lstrip('.')
        if not ext:
            ext = "mp3"

        combined.export(final_output_path, format=ext)
        logger.info(f"Successfully merged clips into {final_output_path}")

    except Exception as e:
        logger.error(f"Error merging audio clips: {e}")
        raise


if __name__ == "__main__":
    # Example usage (can be used for testing)
    # merge_audio_clips(["clip1.mp3", "clip2.mp3"], "output.mp3",
    #                   speakers=["Narrator", "John"])
    pass
