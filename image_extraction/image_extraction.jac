import from byllm.lib {Model, Image}

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

""" Extract memory details from the image, city, date and people."""
def extract_memory_details(
    image: Image, 
    city: str = "",
    date: str = "",
    people: list[str] = []
) -> InitResponse by llm();