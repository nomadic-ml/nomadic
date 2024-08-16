from typing import List, Optional, Dict, Union
from openai import OpenAI
from pydantic import BaseModel, Field

from nomadic.model import DEFAULT_OPENAI_MODEL


from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
import re
import time


class PromptTuner(BaseModel):
    prompting_approaches: List[str] = Field(
        default=["zero-shot", "few-shot", "chain-of-thought"],
        description="List of prompting approaches to use.",
    )
    prompt_complexities: List[str] = Field(
        default=["simple", "complex"],
        description="List of prompt complexities to use.",
    )
    prompt_focuses: List[str] = Field(
        default=["fact extraction", "action points"],
        description="List of prompt focuses to use.",
    )
    enable_logging: bool = Field(
        default=True,
        description="Flag to enable or disable print logging.",
    )

    def generate_prompt_variants(
        self, user_prompt_request: str, api_key: Optional[str] = None
    ) -> List[str]:
        if not api_key:
            raise ValueError("OpenAI API key is required for prompt tuning.")

        client = OpenAI(api_key=api_key)
        prompt_variants = []

        for approach in self.prompting_approaches:
            for complexity in self.prompt_complexities:
                for focus in self.prompt_focuses:
                    if self.enable_logging:
                        print(
                            f"\nGenerating prompt for: Approach={approach}, Complexity={complexity}, Focus={focus}"
                        )

                    system_message = self._create_system_message(
                        user_prompt_request, approach, complexity, focus
                    )

                    try:
                        response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": system_message},
                                {
                                    "role": "user",
                                    "content": "Generate the prompt variant.",
                                },
                            ],
                            temperature=0.7,
                        )

                        generated_prompt = response.choices[0].message.content.strip()

                        # For few-shot approach, generate examples and incorporate them
                        if approach == "few-shot":
                            examples = self._generate_examples(
                                client, user_prompt_request, complexity, focus
                            )
                            generated_prompt = self._incorporate_examples(
                                generated_prompt, examples
                            )

                        prompt_variants.append(generated_prompt)

                        if self.enable_logging:
                            print(
                                f"Generated prompt:\n{generated_prompt[:500]}..."
                            )  # Increased preview length

                    except Exception as e:
                        if self.enable_logging:
                            print(f"Error generating prompt variant: {str(e)}")

        return prompt_variants

    def _create_system_message(
        self, user_prompt_request: str, approach: str, complexity: str, focus: str
    ) -> str:
        base_message = f"""
        You are an AI assistant specialized in generating prompts for various tasks.
        Generate a prompt based on the following parameters:
        - Prompting Approach: {approach}
        - Prompt Complexity: {complexity}
        - Prompt Focus: {focus}

        Use the following user-provided prompt as a basis:

        {user_prompt_request}

        Adjust the prompt based on the specified approach, complexity, and focus:
        """

        approach_instructions = {
            "zero-shot": "Create a prompt that doesn't provide any examples but clearly explains the task and expected output.",
            "few-shot": "Create a prompt that includes placeholder markers for examples. Use [EXAMPLE_1], [EXAMPLE_2], etc. These will be replaced with relevant examples later.",
            "chain-of-thought": "Create a prompt that encourages step-by-step reasoning. Include instructions for the model to explain its thought process and break down complex tasks into smaller steps.",
        }

        complexity_instructions = {
            "simple": "Use straightforward language and keep the instructions concise. Focus on the core task without adding too many details.",
            "complex": "Use more sophisticated language and provide detailed instructions. Include nuanced aspects of the task and potential considerations.",
        }

        focus_instructions = {
            "fact extraction": "Optimize the prompt to emphasize identifying and extracting key factual information from the given context.",
            "action points": "Tailor the prompt to focus on deriving actionable insights or specific steps to be taken based on the information provided.",
        }

        return base_message + "\n".join(
            [
                approach_instructions[approach],
                complexity_instructions[complexity],
                focus_instructions[focus],
                "\nEnsure that the generated prompt variant clearly reflects the specified approach, complexity, and focus.",
            ]
        )

    def _generate_examples(
        self, client: OpenAI, user_prompt_request: str, complexity: str, focus: str
    ) -> List[Dict[str, str]]:
        system_message = f"""
        Based on the following prompt, generate 3 example input-output pairs that would be suitable for few-shot learning.
        The examples should be relevant to the topic and task described in the prompt.
        Adjust the complexity and focus of the examples according to these parameters:
        - Complexity: {complexity}
        - Focus: {focus}

        Prompt: {user_prompt_request}

        Provide the examples in the following format:
        Input: [input text]
        Output: [output text]

        Ensure that the examples are diverse and cover different aspects of the task.
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": "Generate the examples."},
            ],
            temperature=0.7,
        )

        examples_text = response.choices[0].message.content.strip()
        examples = []

        # Parse the examples
        example_pairs = re.findall(
            r"Input: (.*?)Output: (.*?)(?=Input:|$)", examples_text, re.DOTALL
        )
        for input_text, output_text in example_pairs:
            examples.append(
                {"input": input_text.strip(), "output": output_text.strip()}
            )

        return examples

    def _incorporate_examples(self, prompt: str, examples: List[Dict[str, str]]) -> str:
        for i, example in enumerate(examples, 1):
            example_text = f"Example {i}:\nInput: {example['input']}\nOutput: {example['output']}\n\n"
            prompt = prompt.replace(f"[EXAMPLE_{i}]", example_text)
        return prompt

    def update_params(self, params: Dict[str, Any]):
        if "prompt_tuning_approach" in params:
            self.prompting_approaches = [params["prompt_tuning_approach"]]
        if "prompt_tuning_complexity" in params:
            self.prompt_complexities = [params["prompt_tuning_complexity"]]
        if "prompt_tuning_focus" in params:
            self.prompt_focuses = [params["prompt_tuning_focus"]]

    def __str__(self):
        return f"PromptTuner(approaches={self.prompting_approaches}, complexities={self.prompt_complexities}, focuses={self.prompt_focuses})"

    def __repr__(self):
        return self.__str__()


def custom_evaluate(
    response: str,
    evaluation_metrics: List[Dict[str, Union[str, float]]],
    openai_api_key: str,
    max_retries: int = 3,
    retry_delay: int = 5,  # seconds
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

    def get_evaluation_result(prompt):
        completion = client.chat.completions.create(
            model=DEFAULT_OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant tasked with evaluating responses.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content

    def parse_evaluation_result(evaluation_result):
        lines = evaluation_result.split("\n")
        scores = {}
        explanation = ""
        overall_score = 0

        for line in lines:
            if ":" in line:
                parts = line.split(":", 1)  # Split only on the first colon
                if len(parts) == 2:
                    metric, score_str = parts
                    metric = metric.strip()
                    score_str = score_str.strip()
                    try:
                        # Use regular expression to find the first number in the string
                        score_match = re.search(r"\d+(\.\d+)?", score_str)
                        if score_match:
                            score = float(score_match.group())
                            scores[metric] = score
                        else:
                            print(
                                f"Warning: No numeric score found for metric '{metric}'. Score string: '{score_str}'"
                            )
                    except ValueError as e:
                        print(
                            f"Error parsing score for metric '{metric}': {e}. Score string: '{score_str}'"
                        )
            elif line.lower().startswith("overall score:"):
                try:
                    overall_score_match = re.search(
                        r"\d+(\.\d+)?", line.split(":", 1)[1]
                    )
                    if overall_score_match:
                        overall_score = float(overall_score_match.group())
                    else:
                        print(
                            f"Warning: No numeric overall score found. Line: '{line}'"
                        )
                except ValueError as e:
                    print(f"Error parsing overall score: {e}. Line: '{line}'")
            elif line.lower().startswith("brief explanation:"):
                explanation = line.split(":", 1)[1].strip()

        return scores, overall_score, explanation

    retry_count = 0
    while retry_count < max_retries:
        evaluation_result = get_evaluation_result(evaluation_prompt)
        scores, overall_score, explanation = parse_evaluation_result(evaluation_result)

        if all(metric in scores for metric in metrics) and overall_score:
            break  # Exit loop if all metrics are successfully scored

        print(f"Retrying evaluation... ({retry_count + 1}/{max_retries})")
        time.sleep(retry_delay)
        retry_count += 1

    # Calculate weighted score
    if scores:
        total_weight = sum(weights.values())
        weighted_score = (
            sum(scores.get(metric, 0) * weights.get(metric, 1) for metric in metrics)
            / total_weight
        )
    else:
        print("Warning: No valid scores were found. Using overall score if available.")
        weighted_score = overall_score

    return {
        "scores": scores,
        "overall_score": weighted_score,
        "explanation": explanation,
        "evaluation_prompt": evaluation_prompt,
    }
