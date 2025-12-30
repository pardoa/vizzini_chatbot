from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class VizziniChatbot():
    """VizziniChatbot crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def customer_service_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['customer_service_agent'], # type: ignore[index]
            # human_input=True,  # This is the key
            # tracing=True,
            # async_execution=False,
            verbose=True
        )

    # @agent
    # def coffee_beans_specialist(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['coffee_beans_specialist'], # type: ignore[index]
    #         verbose=True
    #     )
    #
    # @agent
    # def order_link_generator(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['order_link_generator'],  # type: ignore[index]
    #         verbose=True
    #     )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def gather_customer_preferences(self) -> Task:
        return Task(
            human_input=True,
            config=self.tasks_config['gather_customer_preferences'], # type: ignore[index]
        )

    # @task
    # def search_and_recommend_blends(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['search_and_recommend_blends'], # type: ignore[index]
    #         output_file='report.md'
    #     )
    #
    # @task
    # def generate_link_and_finalize(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['generate_link_and_finalize'], # type: ignore[index]
    #         output_file='report.md'
    #     )

    @crew
    def crew(self) -> Crew:
        """Creates the VizziniChatbot crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            tracing=True
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
