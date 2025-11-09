import mauve
from typing import List, Tuple, Optional, Union


def calculate_mauve(
        reference_text: str,
        generated_text: str,
        device_id: int = -1,
        max_text_length: Optional[int] = None
) -> float:
    out = mauve.compute_mauve(
        p_text=[reference_text],
        q_text=[generated_text],
        device_id=device_id,
        max_text_length=max_text_length,
        featurize_model_name="gpt2-large"
    )

    # Extract MAUVE score
    mauve_score = out.mauve

    return mauve_score