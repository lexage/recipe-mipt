from autogen_agentchat.agents import (
    AssistantAgent,
    UserProxyAgent,
    GroupChat,
    GroupChatManager
    )


# Define the Planner Agent
planner = AssistantAgent(
    name='Planner',
    system_message='Planner. Suggest a plan. Revise the plan based on feedback from a critic agent.',
    llm_config={"config_list": [{"model": "gpt-4"}]} # Example LLM config
)


# Define the Critic Agent
critic = AssistantAgent(
    name='Critic',
    system_message='Critic. Review the plan provided by the Planner. Provide constructive feedback and identify potential issues or improvements.',
    llm_config={"config_list": [{"model": "gpt-4"}]} # Example LLM config
)


# Define a User Proxy Agent to initiate the task
user_proxy = UserProxyAgent(
    name='User_Proxy',
    human_input_mode='NEVER', # Or 'ALWAYS' for human intervention
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "planning_critic_work_dir"}
)


# Create a GroupChat for managing interactions
groupchat = GroupChat(
    agents=[user_proxy, planner, critic],
    messages=[],
    max_round=10
)


manager = GroupChatManager(
    groupchat=groupchat,
    llm_config={"config_list": [{"model": "gpt-4"}]}
)


# Initiate the conversation
user_proxy.initiate_chat(
    manager,
    message="Develop a marketing strategy for a new eco-friendly product."
)
