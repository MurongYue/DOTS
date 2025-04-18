from utils import *
from src.model_api import select_llm_model
from common import map_with_progress
import argparse

def construct_prompt(query, action):
    action_question = []
    if "planning" in action:
        action_question.append("Why is question decomposition helpful for solving this question?")
    elif "query_rewriting" in action:
        action_question.append("Why is question rewriting helpful for solving this question?")
    if "chain of thought" in action or "CoT" in action:
        action_question.append("Why is using natural language easier to solve this question, and using programming can make a mistake?")
    elif "programming_solver" in action:
        action_question.append("Why is using programming is easier than using natural language for solving this question?")
    if 'verifier' in action:
        action_question.append("Why is verification for this question feasible and important?")
    action_question_string = "\n".join(action_question)
    action_string = ",".join(action)
    prompt = f"""Action Categories:
1. Understanding process:
query_rewriting: Rewrite the question and answer it.
planning: Decompose the questions into multiple subtasks solve the sub-question.
2. Solving process:
chain of thought: For step-by-step reasoning with language.
programming: For programming solver.
3. Verification process:
verifier: To check the correctness of the solution.


Query: 
Find $2 \cdot 5^{{-1}} + 8 \cdot 11^{{-1}} \pmod{{56}}$.\n\nExpress your answer as an integer from $0$ to $55$, inclusive.
Task Instruction: For the given question, explain following questions in one paragraph:
Why using programming is easy than using natural language for solving this question?
Why verification for this question is feasible and important?
Explanation: This problem involves modular arithmetic and finding modular inverses, which can be efficiently solved using a programming approach. Programming is better than a chain of thought (CoT) in this case because it allows for precise calculations and handling of potential edge cases that might be cumbersome to address manually. A program can quickly compute the modular inverses and perform the arithmetic operations accurately. Verification is important to ensure the correctness of the solution, as errors in calculating modular inverses or arithmetic operations can lead to incorrect results. By verifying the solution, we can confirm that the program has correctly implemented the mathematical operations and that the final answer is within the specified range.

Query:
What is the largest four-digit number whose digits add up to 16?
Task Instruction: For the given question, explain following questions in one paragraph:
Why using programming is easy than using natural language for solving this question?
Why verification for this question is feasible and important?
Explanation: This problem involves finding the largest four-digit number whose digits sum up to 16. Using a programming approach is more efficient than a chain of thought (CoT) because it allows for systematic exploration of all possible combinations of four-digit numbers and their digit sums. A program can quickly iterate through potential candidates, calculate their digit sums, and identify the largest number that meets the criteria.  Verification is important in this context to ensure that the program's output is correct. By verifying the solution, we can confirm that the program has correctly identified the largest number and that the digit sum condition is satisfied. 

Query: 
Let $f(x) = x^2-3x$. For what values of $x$ is $f(f(x)) = f(x)$? Enter all the solutions, separated by commas.
Task Instruction: For the given question, explain following questions in one paragraph:
Why using programming is worse than using natural language for this question?
Why verification for this question is feasible and important?
Explanation: This problem involves solving the equation \( f(f(x)) = f(x) \) where \( f(x) = x^2 - 3x \). Using a chain of thought approach is more suitable than programming because it allows for symbolic manipulation and algebraic reasoning, which are essential for understanding the behavior of the function and solving the equation analytically. Programming might not be straightforward here because it would require implementing symbolic computation or numerical methods, which can be complex and less intuitive for this type of problem. Verification is important to ensure that the solutions obtained are correct and satisfy the original equation. By verifying, we confirm that the solutions are not only mathematically valid but also consistent with the problem's requirements.

Query: 
What is the value of $a$ for which $\\frac{{1}}{{\\text{{log}}_2a}} + \\frac{{1}}{{\\text{{log}}_3a}} + \\frac{{1}}{{\\text{{log}}_4a}} = 1$?
Task Instruction: For the given question, explain following questions in one paragraph:
WWhy is using natural language easier to solve this question, and using programming can make a mistake?
Explanation: This problem involves solving an equation with logarithms, which requires algebraic manipulation and understanding of logarithmic properties. Using a programming approach is not straightforward because it involves symbolic computation and solving equations analytically, which is more naturally handled through mathematical reasoning rather than numerical computation. Chain of thought (CoT) is appropriate here as it allows for step-by-step reasoning to simplify and solve the equation. By breaking down the problem into smaller parts, we can systematically apply logarithmic identities and algebraic techniques to find the value of \(a\). This approach ensures a clear understanding of the solution process and the mathematical principles involved.

Query:
{query}
Task Instruction: For the given question, explain following questions in one paragraph:
{action_question_string} 
Explanation:
"""
    return prompt


def get_action_accuracy(action, searching_result):
    correct = 0
    wrong = 0
    correct_wo = 0
    wrong_wo = 0
    for d in searching_result:
        if action in d['trajectory']:
            if d['score'] == 1:
                correct += 1
            else:
                wrong += 1
        else:
            if d['score'] == 1:
                correct_wo += 1
            else:
                wrong_wo += 1
    return correct / (correct + wrong), correct_wo / (correct_wo + wrong_wo), correct + wrong, correct_wo + wrong_wo


def select_important_actions(d, importance_score=0.3):
    global new_data
    global not_searched_num
    searching_result = d['searching_results']
    planning_accuracy = get_action_accuracy(action='planning', searching_result=searching_result)
    query_rewriting_accuracy = get_action_accuracy(action='query_rewriting', searching_result=searching_result)
    CoT_accuracy = get_action_accuracy(action='CoT', searching_result=searching_result)
    programming_solver_accuracy = get_action_accuracy(action='programming_solver', searching_result=searching_result)
    verification_accuracy = get_action_accuracy(action='verifier', searching_result=searching_result)
    important_actions = []
    action_accuracies = [planning_accuracy, query_rewriting_accuracy, CoT_accuracy, programming_solver_accuracy,
                         verification_accuracy]
    action_names = ["planning", "query_rewriting", "CoT", "programming_solver", "verifier"]
    for var, name in zip(action_accuracies, action_names):
        if var[0] > var[1] + importance_score and abs(var[2] - var[3]) < 0.6 * (var[2] + var[3]):
            important_actions.append(name)
    meta_info = d['meta_info']
    if len(important_actions) > 0:
        correct_solution = ''
        corresponding_action_trajectory = []
        if 'planning' in important_actions:
            corresponding_action_trajectory.append('planning')
        elif 'query_rewriting' in important_actions:
            corresponding_action_trajectory.append('query_rewriting')
        else:
            corresponding_action_trajectory.append('')
        if 'programming_solver' in important_actions:
            corresponding_action_trajectory.append('programming_solver')
        else:
            corresponding_action_trajectory.append('CoT')
        if 'verifier' in important_actions:
            corresponding_action_trajectory.append('verifier')
        else:
            corresponding_action_trajectory.append('')
        corresponding_action_trajectory.append("direct_answering")
        for s in searching_result:
            if s['trajectory'] == corresponding_action_trajectory and s['score'] == 1:
                correct_solution = s['reasoning dialogues']
        if len(correct_solution)>0:
            new_data.append({'query': meta_info['problem'], 'action': important_actions, 'solution': correct_solution})
        else:
            not_searched_num+=1

def explanation_generation(d):
    query = d['query']
    action = d['action']
    prompt = construct_prompt(query=query, action=action)
    explanation = llm.invoke(prompt=prompt, temperature=0.0)
    d['explanation'] = explanation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--input_file", type=str,
                        default="./data/llama3_8b_searched.json")
    parser.add_argument("--output_file", type=str,
                        default="./data/math_explanation.json")
    parser.add_argument("--debug", type=lambda x: x.lower() == 'true', default="true")
    args = parser.parse_args()
    data = read_json(args.input_file)
    if args.debug:
        data = data[:10]
    new_data = []
    not_searched_num = 0
    llm = select_llm_model(llm_model=args.api_model, host='')
    for d in data:
        select_important_actions(d)
    print("Explanation Generating")
    map_with_progress(explanation_generation, new_data, 50)
    save_json(filename=f"{args.output_file}", data=new_data)

