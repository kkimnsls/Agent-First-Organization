import logging
from functools import partial
from langgraph.graph import StateGraph, START

from arklex.env.workers.worker import BaseWorker, register_worker
from arklex.utils.graph_state import MessageState
from arklex.env.tools.utils import ToolGenerator
from arklex.env.tools.RAG.retrievers.milvus_retriever import RetrieveEngine


logger = logging.getLogger(__name__)


@register_worker
class MilvusRAGWorker(BaseWorker):

    description = "Answer the user's questions based on the company's internal documentations (unstructured text data), such as the policies, FAQs, and product information"

    def __init__(self,
                 # stream_ reponse is a boolean value that determines whether the response should be streamed or not.
                 # i.e in the case of RagMessageWorker it should be set to false.
                 stream_response: bool = True):
        super().__init__()
        self.stream_response = stream_response

    def choose_tool_generator(self, state: MessageState):
        if self.stream_response and state.is_stream:
            return "stream_tool_generator"
        return "tool_generator"

    def _create_action_graph(self, tags: dict):
        workflow = StateGraph(MessageState)
        # Create a partial function with the extra argument bound
        retriever_with_args = partial(RetrieveEngine.milvus_retrieve, tags=tags)
        # Add nodes for each worker
        workflow.add_node("retriever", retriever_with_args)
        workflow.add_node("tool_generator", ToolGenerator.context_generate)
        workflow.add_node("stream_tool_generator", ToolGenerator.stream_context_generate)
        # Add edges
        workflow.add_edge(START, "retriever")
        workflow.add_conditional_edges(
            "retriever", self.choose_tool_generator)
        return workflow

    def _execute(self, msg_state: MessageState, **kwargs):
        self.tags : dict = kwargs.get("tags", {})
        self.action_graph = self._create_action_graph(self.tags)
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        return result
