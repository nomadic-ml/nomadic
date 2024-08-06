from typing import List, Optional
from openai import OpenAI
from pydantic import BaseModel, Field


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

        system_message = f"""
        Generate {self.num_iterations_per_prompt} prompt variants based on the following parameters:
        - Prompting Approach: {approach}
        - Prompt Complexity: {complexity}
        - Prompt Focus: {focus}

        Each prompt should be a variation of the original prompt, adhering to the specified parameters.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": f"Original prompt: {prompt}",
                    },
                ],
                max_tokens=200,
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


def custom_evaluate(response: str, evaluation_metrics: List[str]) -> dict:
    metrics_prompt = "\n".join(
        [f"{i+1}. {metric}" for i, metric in enumerate(evaluation_metrics)]
    )
    metrics_format = "\n".join([f"{metric}: [score]" for metric in evaluation_metrics])

    evaluation_prompt = f"""
    You are a judge evaluating the quality of a generated response for a financial advice summary.
    Please evaluate the following response based on these criteria, scoring each from 0 to 20:

    {metrics_prompt}

    Response to evaluate:
    {response}

    Provide your evaluation in the following format:
    {metrics_format}
    Overall score: [total out of {len(evaluation_metrics) * 20}]

    Brief explanation: [Your explanation of the strengths and weaknesses]
    """

    return {"evaluation_prompt": evaluation_prompt}
