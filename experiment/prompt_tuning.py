from typing import List, Optional, Dict
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

    def _generate_zero_shot_prompt(self, prompt: str) -> str:
        return f"""
        Given the following task, provide a direct response without any examples or additional context:

        Task: {prompt}
        """

    def _generate_few_shot_prompt(self, prompt: str) -> str:
        return f"""
        Here are a few examples of how to approach similar tasks:

        Example 1:
        Task: Summarize the key points of a financial report.
        Response: 1. Review revenue and profit figures. 2. Analyze growth trends. 3. Examine debt levels and cash flow. 4. Highlight major business segments. 5. Note any significant events or changes.

        Example 2:
        Task: Explain the concept of diversification in investing.
        Response: Diversification involves spreading investments across various assets to reduce risk. It can include different asset classes, industries, and geographic regions.

        Now, please address the following task:

        Task: {prompt}
        """

    def _generate_chain_of_thought_prompt(self, prompt: str) -> str:
        return f"""
        For the following task, provide a detailed step-by-step solution, showing your reasoning at each stage:

        Task: {prompt}

        Your response should follow this structure:
        1. Initial thoughts
        2. Breaking down the problem
        3. Step-by-step solution
        4. Final answer or conclusion
        5. Verification or sanity check
        """

    def _generate_task_decomposition_prompt(self, prompt: str) -> str:
        return f"""
        For the following task, break it down into smaller, manageable subtasks:

        Main Task: {prompt}

        Your response should follow this structure:
        1. List of subtasks
        2. Brief explanation of each subtask
        3. Suggested order of completion
        4. Any dependencies between subtasks
        """

    def _generate_self_consistency_prompt(self, prompt: str) -> str:
        return f"""
        Please provide multiple different approaches or solutions to the following task:

        Task: {prompt}

        Your response should include:
        1. At least three different approaches or solutions
        2. A brief explanation of each approach
        3. A comparison of the strengths and weaknesses of each approach
        4. A final recommendation on the best approach and why
        """

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
                max_tokens=300,
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


from openai import OpenAI
import os
from typing import List


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

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
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
    overall_score = 0
    explanation = ""

    for line in lines:
        if ":" in line:
            metric, score = line.split(":")
            metric = metric.strip()
            try:
                score = int(score.strip().split()[0])  # Extract the numeric score
                scores[metric] = score
            except ValueError:
                pass  # Skip lines that don't contain a numeric score
        elif line.lower().startswith("overall score:"):
            try:
                overall_score = int(line.split(":")[1].strip().split()[0])
            except ValueError:
                pass
        elif line.lower().startswith("brief explanation:"):
            explanation = line.split(":", 1)[1].strip()
    return {
        "scores": scores,
        "overall_score": overall_score,
        "explanation": explanation,
        "evaluation_prompt": evaluation_prompt,
    }
