import from fastapi { HTTPException, status }
import from jac_cloud.jaseci.utils.logger { logger }
import from nodes_and_edges { memories, memory, comment, profile, shared_memory }
import from jaseci_sdk { get_node_owner, get_node_by_root_id }
import from profile { visit_profile }
import from mtp { generate_comment_summary_and_povs, search_memories_from_query }
import from ai_client { openai_client }

walker visit_memories {

    class __specs__ {
        has tags: list = ["memory"];
    }
    
    can visit_memories with `root entry;
}

walker visit_memory(visit_memories) {
    has memory_id: str;

    class __specs__ {
        has tags: list = ["memory"];
    }

    can visit_memory with memories entry;
}

walker get_memory(visit_memories) {
    has memory_id: str;

    class __specs__ {
        has tags: list = ["memory"];
    }

    can check_memories with memories entry;
    can check_shared_memories with profile entry;
    can get_memory with memory entry;
}

walker list_memories(visit_memories) {

    class __specs__ {
        has tags: list = ["memory"];
    }

    can list_memories with memories entry;
}

walker search_memories(visit_memories) {
    has query: str;
    has use_byllm: bool = True;

    class __specs__ {
        has tags: list = ["memory"];
    }

    can search_memories with memories entry;
}

walker migrate_profile_data(visit_memories) {
    has json_file_content:list;


    can migrate_profile_data_from_json with memories entry {
        if len(self.json_file_content) == 0 {
            
            return;
        }

        data = self.json_file_content;

        for item in data {
            data_block = item["context"];
            memory_node = memory(
                memory_id = data_block["memory_id"],
                session_id = data_block["session_id"],
                summary = data_block["summary"],
                when = data_block["when"][0],
                who = data_block["who"],
                where = data_block["where"],
                what = data_block["what"][0],
                created_at = data_block["created_at"],
                updated_at = data_block["updated_at"],
                image_url = " ",
                conversation = data_block["conversation"]
            
            );

            here ++> memory_node;
        }

        # print("----------------------->");
        # print([-->](`?memory));
        # report {};

        
    }
}