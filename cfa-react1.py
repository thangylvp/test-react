import langchain
# langchain.debug = True
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.tools import BaseTool
from langchain.agents import initialize_agent
from db.memory import retrieval
from langchain.load.dump import dumps
import sympy
import argparse
import json
import os 
import yaml
from tqdm import tqdm 
from dataset.cfal1 import CfaQADataset

class scientific_calculator(BaseTool):
    name = "Scientific calculator"
    description = """
        "usage": This is a scientific calculator, use for calculate math expressions.
        "input": The input is string represents a list of string, each item represents a math expression starts with \\" and ends with \\". Do not use comma , to represent numbers.
        "example input": "[\\"2 * (4 ^(1/2))\\"]"
    """
    def _run(self, data):
        return sympy.sympify(data)
    
class search_knowledge(BaseTool):
    name = "Search knowledge"
    description = """
        "usage": Use this tool when you need to search for something. This tool returns 5 related knowledge to your query.
        "input": The input is a string represents your query starts with \\" and ends with \\", wrap in a list.
        "example input": "[\\"What is CFA?\\"]"
    """
    def _run(self, data):
        related_context = retrieve_db.query(data[0], top_k=5)
        related_context = [f"{p},\n\n" for p in related_context]
        related_context = ''.join(related_context)
        related_context = f"[{related_context}]"
        return related_context

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cfg", default='configs', type=str)

    args = parser.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    retrieve_db = retrieval(cfg=cfg)

    llm = ChatOpenAI(
        temperature=0.2,
        model_name= "gpt-3.5-turbo-0125" # "gpt-4-turbo-preview" # 'gpt-3.5-turbo-0125' #'gpt-4'
    )

    # initialize conversational memory
    
    cfa = CfaQADataset(cfg)    
    for name in tqdm(cfa.questions):
        # print()
        # print()
        # print(name)
        # print()
        # print()
        mock = cfa.map_q[name]
        if mock != "ex1":
            continue
        if os.path.exists(os.path.join("log_react1", f"{name}.json")):
            continue
        try:
            question = cfa.questions[name]
            
            conversational_memory = ConversationBufferWindowMemory(
                memory_key='chat_history',
                k=5,
                return_messages=True,
                input_key='input', 
                output_key="output"
            )

            tools = [scientific_calculator(), search_knowledge()]

            agent = initialize_agent(
                agent='chat-conversational-react-description',
                tools=tools,
                llm=llm,
                verbose=True,
                max_iterations=8,
                early_stopping_method='generate',
                memory=conversational_memory,
                return_intermediate_steps=True
            )
            agent.agent.llm_chain.verbose=True    
            
            # print(mock)
            query = f"{question['question']}\n{question['choices']}"
            output = agent(query)
            # with open(os.path.join("log_react1", f"{name}.json"), "w") as f:
            #     f.write(dumps(output))
        except:
            pass
        # break

    # print()
    # print()
    # print("+" * 200)
    # print(output)
    # print()
    # print()
    # print(conversational_memory)

