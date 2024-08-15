from typing import List, Optional, Dict, Union
from openai import OpenAI
from pydantic import BaseModel, Field

from nomadic.model import DEFAULT_OPENAI_MODEL


class PromptTuner(BaseModel):
    num_iterations_per_prompt: int = Field(
        default=3,
        description="Number of iterations per prompt for generating variants.",
    )
    prompting_approaches: List[str] = Field(
        default=["few-shot"],
        description="List of prompting approaches to use.",
    )
    prompt_complexities: List[str] = Field(
        default=["simple"],
        description="List of prompt complexities to use.",
    )
    prompt_focuses: List[str] = Field(
        default=["general"],
        description="List of prompt focuses to use.",
    )
    enable_logging: bool = Field(
        default=True,
        description="Flag to enable or disable print logging.",
    )

    def generate_similar_prompts(
        self,
        prompt: str,
        approach: str,
        complexity: str,
        focus: str,
        api_key: Optional[str] = None,
    ) -> List[str]:
        if not api_key:
            return [prompt]

        client = OpenAI(api_key=api_key)

        approach_prompt = getattr(
            self, f"_generate_{approach.replace('-', '_')}_prompt"
        )(prompt)

        system_message = f"""
        Generate {self.num_iterations_per_prompt} prompt variants based on the following parameters:
        - Prompting Approach: {approach}
        - Prompt Complexity: {complexity}
        - Prompt Focus: {focus}

        Use the following approach-specific prompt as a basis:

        {approach_prompt}

        Adjust the complexity and focus of the prompts accordingly:
        - Complexity: {complexity} (adjust language and concept difficulty)
        - Focus: {focus} (emphasize specific aspects or outcomes)
        """

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Original prompt: {prompt}"},
                ],
                n=self.num_iterations_per_prompt,
                stop=None,
                temperature=0.7,
            )

            prompt_variants = [
                choice.message.content.strip() for choice in response.choices
            ]
            return prompt_variants
        except Exception as e:
            if self.enable_logging:
                print(f"Error generating similar prompts: {str(e)}")
            return [prompt]

    def generate_prompt_variants(
        self, user_prompt_request: str, api_key: Optional[str] = None
    ) -> List[str]:
        prompt_variants = []
        for approach in self.prompting_approaches:
            for complexity in self.prompt_complexities:
                for focus in self.prompt_focuses:
                    if self.enable_logging:
                        print(
                            f"\nGenerating prompts for: Approach={approach}, Complexity={complexity}, Focus={focus}"
                        )
                    generated_prompts = self.generate_similar_prompts(
                        user_prompt_request, approach, complexity, focus, api_key
                    )
                    prompt_variants.extend(generated_prompts)
        return prompt_variants

    def update_params(self, params: Dict):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if self.enable_logging:
            print(f"Updated PromptTuner parameters: {params}")


def custom_evaluate(
    response: str,
    evaluation_metrics: List[Dict[str, Union[str, float]]],
    openai_api_key: str,
) -> dict:
    metrics = [metric["metric"] for metric in evaluation_metrics]
    weights = {metric["metric"]: metric["weight"] for metric in evaluation_metrics}

    metrics_prompt = "\n".join([f"{i+1}. {metric}" for i, metric in enumerate(metrics)])
    metrics_format = "\n".join([f"{metric}: [score]" for metric in metrics])

    evaluation_prompt = f"""
    You are a judge evaluating the quality of a generated response for a financial advice summary.
    Please evaluate the following response based on these criteria, scoring each from 0 to 20:

    {metrics_prompt}

    Response to evaluate:
    {response}

    Provide your evaluation in the following format:
    {metrics_format}
    Overall score: [total out of {len(metrics) * 20}]

    Brief explanation: [Your explanation of the strengths and weaknesses]
    """

    client = OpenAI(api_key=openai_api_key)
    completion = client.chat.completions.create(
        model=DEFAULT_OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant tasked with evaluating responses.",
            },
            {"role": "user", "content": evaluation_prompt},
        ],
    )

    evaluation_result = completion.choices[0].message.content

    # Parse the evaluation result
    lines = evaluation_result.split("\n")
    scores = {}
    explanation = ""

    for line in lines:
        if ":" in line:
            parts = line.split(":", 1)  # Split only on the first colon
            if len(parts) == 2:
                metric, score_str = parts
                metric = metric.strip()
                try:
                    score = float(
                        score_str.strip().split()[0]
                    )  # Extract the numeric score
                    scores[metric] = score
                except ValueError:
                    pass  # Skip lines that don't contain a numeric score
        elif line.lower().startswith("overall score:"):
            try:
                overall_score = float(line.split(":", 1)[1].strip().split()[0])
            except ValueError:
                pass
        elif line.lower().startswith("brief explanation:"):
            explanation = line.split(":", 1)[1].strip()

    # Calculate weighted score
    total_weight = sum(weights.values())
    weighted_score = (
        sum(scores.get(metric, 0) * weights.get(metric, 1) for metric in metrics)
        / total_weight
    )

    return {
        "scores": scores,
        "overall_score": weighted_score,
        "explanation": explanation,
        "evaluation_prompt": evaluation_prompt,
    }
