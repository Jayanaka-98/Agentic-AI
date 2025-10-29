import from byllm.lib {Model, Image}
import from ai_client { openai_client }

glob llm = Model(model_name="gpt-4o-mini");

obj Response {
    has follow_up_questions: str;
    has summary: str;
    has when: str;
    has who: list[str];
    has what: str;
}

obj InitResponse(Response) {
    has location_type: str;
}

sem InitResponse = "Memory details inferred from the image, city, date, and people.";
sem InitResponse.follow_up_questions = "Ask one follow-up question to continue the conversation.";
sem InitResponse.summary = "A concise summary of the memory.";
sem InitResponse.when = "The date of the memory in YYYY-MM-DD format.";
sem InitResponse.who = "Exact names of people in the memory (e.g., [John, Kate]); return [] if none.";
sem InitResponse.what = "What the memory is about.";
sem InitResponse.location_type = "Location type inferred from the image (e.g., waterfall, restaurant).";

""" Extract memory details from the image, city, date and people."""
def extract_memory_details(
    image: Image, 
    city: str = "",
    date: str = "",
    people: list[str] = []
) -> InitResponse by llm();