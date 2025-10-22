import from byllm { Model, Image }
import from typing { List, Dict, Any }

glob llm = Model(model_name="gpt-4.1");

obj Response {
    has follow_up_questions: str;
    has summary: str;
    has when: str;
    has who: List[str];
    has what: str;
}

obj InitResponse(Response) {
    has location_type: str;
}

obj ContinueResponse(Response) {
    has where: List[str];
    has save_memory: bool;
    has show_summary: bool;
}

obj CommentPOVs {
    has email: str;
    has pov: str;
}

obj CommentSummaryPOVs {
    has comment_summary: str;
    has povs: List[CommentPOVs];
}

obj IDResponse {
    has ids: List[int];
}

""" Extract memory details from the image, city, date and people."""
def extract_memory_details(
    image: Image, 
    city: str = "",
    date: str = "",
    people: List[str] = []
) -> InitResponse by llm();

""" Update memory details based on user input and context."""
def update_memory_details(
    image: Image, 
    utterance: str,
    summary: str = "",
    when: str = "",
    who: List[str] = [],
    where: List[str] = [],
    what: str = "",
    conversation: List[dict] = [],
    show_summary: bool = False,
    save_memory: bool = False
) -> ContinueResponse by llm();

""" Generate a summary and points of view for each comment."""
def generate_comment_summary_and_povs(
    memory_summary: str,
    comments: Dict[str, Any]
) -> CommentSummaryPOVs by llm();

""" Search for memories matching the query."""
def search_memories_from_query(
    memories: List[Dict[str, str]], 
    query: str
) -> IDResponse by llm();