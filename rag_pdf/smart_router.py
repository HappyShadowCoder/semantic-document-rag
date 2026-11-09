from typing import Tuple, Any, Optional
from .llm_router import route_to_llm

class SmartRouter:
    """
    Intelligent router that classifies incoming queries by complexity
    and routes to the active model or tier.
    """
    def __init__(self, selected_model: str = "Auto (Fallback Priority)", openai_key: str = None, gemini_key: str = None):
        try:
            self.primary_llm, self.llm_name = route_to_llm(
                selected_model=selected_model,
                custom_openai_key=openai_key,
                custom_gemini_key=gemini_key
            )
        except Exception:
            self.primary_llm = None
            self.llm_name = "Not Configured"

    def update_model(self, selected_model: str, openai_key: str = None, gemini_key: str = None):
        self.primary_llm, self.llm_name = route_to_llm(
            selected_model=selected_model,
            custom_openai_key=openai_key,
            custom_gemini_key=gemini_key
        )

    def route_query(self, query: str) -> Tuple[Optional[Any], str, str]:
        """
        Evaluates query complexity:
        - 'simple': factual lookups, single entity questions, definitions
        - 'complex': multi-hop reasoning, comparison, summarization, analysis
        Returns: (llm_instance, llm_name, complexity_category)
        """
        lower_q = query.lower()
        complex_signals = [
            "compare", "contrast", "analyze", "synthesize", "explain how",
            "difference between", "summarize", "implications", "pros and cons",
            "relationship between", "evaluate"
        ]

        is_complex = any(signal in lower_q for signal in complex_signals) or len(query.split()) > 25
        complexity = "complex" if is_complex else "simple"

        return self.primary_llm, self.llm_name, complexity
