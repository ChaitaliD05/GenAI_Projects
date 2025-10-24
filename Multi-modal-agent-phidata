from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.task import TaskGraph

# Define Specialized Agents
research_agent = Agent(
    name="ResearchAgent",
    role="Web researcher that finds and summarizes relevant information.",
    model=OpenAIChat(model="gpt-4o"),
)

image_agent = Agent(
    name="ImageAgent",
    role="Analyzes image content and extracts visual insights.",
    model=OpenAIChat(model="gpt-4o-mini"),
)

data_agent = Agent(
    name="DataAgent",
    role="Queries structured data (SQL, APIs) and summarizes results.",
    model=OpenAIChat(model="gpt-4o"),
)

#  Create a Controller Agent
controller_agent = Agent(
    name="ControllerAgent",
    role="Determines which sub-agents to invoke based on user intent.",
    model=OpenAIChat(model="gpt-4o"),
)

#  Build an Agentic Graph (workflow)
graph = TaskGraph(name="MultiModalAgentGraph")

graph.add_task("classify_intent", controller_agent)
graph.add_task("research", research_agent, depends_on="classify_intent")
graph.add_task("analyze_image", image_agent, depends_on="classify_intent")
graph.add_task("query_data", data_agent, depends_on="classify_intent")

# Run Orchestration
result = graph.run(input="Analyze this image and find related data trends.")
print(result)
