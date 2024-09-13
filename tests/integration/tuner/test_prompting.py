
import os

# Import relevant Nomadic libraries
from nomadic.experiment import Experiment
from nomadic.model import OpenAIModel
from nomadic.tuner import tune
import pandas as pd
pd.set_option('display.max_colwidth', None)

prompt_template = """
You are an AI assistant specialized in detecting hallucinations in text responses. Your task is to analyze the given query, context, and response, and determine if the response contains any hallucinations or unfaithful information.
Instructions:
1. Carefully read the query, context, and response.
2. Compare the information in the response to the provided context.
3. Identify any statements in the response that are not supported by or contradict the context.
4. Determine if the response is faithful to the query and context, or if it contains hallucinations.
Provide only your judgment as one of the following (just one word):
- "Faithful"
- "Not faithful"
- "Refusal"
Do not provide any additional explanation or analysis.
Query: [QUERY]
Context: [CONTEXT]
Response to evaluate: [RESPONSE]
Your one-word judgment:
"""

evaluation_dataset = [
 {'query': "Authorization code issue like many of the other questions posted on here ... I'm having trouble authenticating my SPSS authorization code. Then there is an issue getting a license code back from the IBM proxy server to complete my download. Please help.",
  'context': 'IBM SPSS Student Version and Graduate Pack Resources - United States SPSS Support Acquisition Statistics Stats Grad Pack Student TECHNOTE (FAQ)\n\nQUESTION\n Where do I get support for IBM SPSS Student Version or Graduate Pack software? \n\nANSWER\nStep 1: \nFor installation & licensing issues on Student version and Graduate pack, contact your vendor. \n\n * Hearne [http://www.hearne.software/Software/SPSS-Grad-Packs-for-Students-by-IBM/FAQ] \n * On the Hub [http://onthehub.com/] \n * StudentDiscounts.com [http://studentdiscounts.com/contact-us.aspx] \n * JourneyEd [https://www.journeyed.com/contact] \n * thinkEDU [https://thinkedu.desk.com/] \n * Studica [http://www.studica.com/technical-support]\n\n\n\n\n\nCross reference information Segment Product Component Platform Version Edition Business Analytics SPSS Amos Windows Not Applicable Grad Pack, Shrinkwrap, Campus Business Analytics SPSS Modeler Platform Independent Not Applicable Business Analytics SPSS Text Analytics for Surveys Windows Not Applicable Business Analytics SPSS Statistics Platform Independent Not Applicable Campus Edition, Grad Pack, Student Version',
  'response': 'I\'m sorry for any confusion, but the text you\'ve provided is not related to the issue you\'re having with SPSS authorization codes and IBM license servers. The text is an excerpt from a document about the differences between UML and IBM Rational Rhapsody, which are tools used for software modeling and design.\n\nFor your issue with SPSS authorization codes and IBM license servers, I\'d be happy to help you troubleshoot. Here are some steps you can take:\n\n1. **Check your authorization code**: Ensure that you\'ve entered your SPSS authorization code correctly. It\'s case-sensitive and should be entered exactly as it appears on your SPSS software or receipt.\n\n2. **Check your internet connection**: The IBM license server needs to communicate with your computer to generate a license code. Make sure your internet connection is stable.\n\n3. **Check your firewall settings**: Sometimes, firewall settings can block the communication between your computer and the IBM license server. You might need to temporarily disable your firewall or add an exception for the IBM license server.\n\n4. **Try a different browser or computer**: If possible, try using a different web browser or computer to rule out any software-specific issues.\n\n5. **Contact IBM Support**: If none of the above steps work, I would recommend reaching out to IBM\'s official support channels. They should be able to help you resolve the issue with your license code.\n\nHere\'s how you can contact IBM Support:\n\n- Go to the IBM Support website: https://www.ibm.com/support/home\n- Select "Software Products" and then "SPSS Statistics"\n- Follow the prompts to create a support ticket or chat with a support representative.',
  'answer': 'Faithful'},
 {'query': 'Highlight the parts (if any) of this contract related to "Non-Transferable License" that should be reviewed by a lawyer. Details: Does the contract limit the ability of a party to transfer the license being granted to a third party?',
  'context': '"2. ENDORSEMENT AND GRANT OF RIGHTS. During the Contract Period: ... c) The License Rights are exclusive to Berkshire and may not be assigned or in any way conveyed by Berkshire without Auriemma\'s express written consent, except in the event of a merger by Berkshire with another entity offering Banking Services."\n\nThis excerpt highlights the non-transferability of the license granted to Berkshire. The license is exclusive to Berkshire and cannot be assigned or conveyed without Auriemma\'s express written consent, with the exception of a merger with another entity offering banking services. This means that Berkshire cannot transfer the license to a third party without Auriemma\'s approval, limiting Berkshire\'s ability to transfer the license.',
  'response': 'Based on the provided excerpt, here are the key parts of the contract related to "Non-Transferable License" that a lawyer should review:\n\n1. **"The License Rights are exclusive to Berkshire"**: This sentence clearly states that the license is exclusive to Berkshire, which means it cannot be used by any other party without Berkshire\'s permission.\n\n2. **"may not be assigned or in any way conveyed by Berkshire without Auriemma\'s express written consent"**: This is the core provision that makes the license non-transferable. It prohibits Berkshire from assigning or conveying the license to any third party without Auriemma\'s explicit, written agreement. A lawyer would need to ensure that this provision aligns with the parties\' intentions and local laws.\n\n3. **"except in the event of a merger by Berkshire with another entity offering Banking Services"**: This is an exception to the non-transferability rule. It allows Berkshire to transfer the license in the event of a merger with another entity that offers banking services. A lawyer would need to review this exception to ensure it is clear, reasonable, and does not inadvertently create other issues or liabilities.\n\nHere are some details a lawyer would consider while reviewing these parts:\n\n- **Clarity and Ambiguity**: Ensure the language is clear and unambiguous to avoid potential disputes. For instance, the term "conveyed" could be interpreted broadly, so it might be helpful to specify the types of transfers prohibited (e.g., assignment, sublicense, etc.).\n\n- **Compliance with Local Laws**: The lawyer should ensure that the non-transferability provision complies with applicable laws, such as those governing intellectual property, contracts, and business entities.\n\n- **Reasonableness and Enforceability**: The lawyer should assess whether the non-transferability provision is reasonable and enforceable. For example, a court might refuse to enforce a provision that is overly broad or restrictive.\n\n- **Consequences of Non-Compliance**: The lawyer should review any penalties or remedies for breach of the non-transferability provision to ensure they are fair and reasonable.\n\n- **Interaction with Other Provisions**: The lawyer should ensure that the non-transferability provision does not conflict with other provisions in the contract, such as those related to change of control, assignment, or termination.\n\nIn summary, a lawyer should review these parts of the contract to ensure they accurately reflect the parties\' intentions, comply with applicable laws, and are clear, reasonable, and enforceable.',
  'answer': 'Faithful'},
 {'query': 'Does rugby headgear prevent concussion?',
  'context': 'To examine the attitudes of players and coaches to the use of protective headgear, particularly with respect to the prevention of concussion. A questionnaire designed to assess attitudes to headgear was administered to 63 players from four different Canadian teams, each representing a different level of play (high school, university, community club, national). In addition, coaches from all four levels were questioned about team policies and their personal opinions about the use of headgear to prevent concussion. Although the players tended to believe that the headgear could prevent concussion (62%), the coaches were less convinced (33%). Despite the players\' belief that headgear offers protection against concussion, only a minority reported wearing headgear (27%) and few (24%) felt that its use should be made mandatory. Common reasons for not wearing headgear were "its use is not mandatory", "it is uncomfortable", and "it costs too much".',
  'response': 'Under the conditions of this study, pulpotomy offered a viable alternative to root canal treatment for teeth with vital pulps in the short term. However, there is insufficient clinical evidence to consider this technique for the treatment of every permanent tooth. Nevertheless, it should be considered as a potential alternative approach to be further developed for future applications.',
  'answer': 'Not faithful'}]

temperature_search_space = tune.choice([0.1,0.9])
max_tokens_search_space = tune.choice([25, 100])
prompt_tuning_approach = tune.choice(["selfcheck-prompt"])
prompt_tuning_complexity = tune.choice(["simple"])

experiment = Experiment(
    params={"temperature","max_tokens", "prompt_approach","prompt_complexity"},
    user_prompt_request=prompt_template,
    model=OpenAIModel(model="gpt-4", api_keys={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]}),
    evaluator={"method": "custom_evaluate_hallucination"},
    search_method="grid",
    enable_logging=True,  # Set to True for debugging, False for production
    use_flaml_library=False,
    name="Hallucination Detection Experiment",
    evaluation_dataset=evaluation_dataset,
    num_simulations = 1,
)

experiment_result = experiment.run(
    param_dict={
        "temperature": temperature_search_space,
        "prompt_tuning_approach": prompt_tuning_approach,
        "prompt_tuning_complexity": prompt_tuning_complexity,
    }
)

exit()
